import os
import torch
import yaml
import torchaudio
import librosa
import warnings
from numpy import ndarray
from numpy import float32


warnings.simplefilter('ignore')

# Import necessary modules (assuming they are in the same directory or properly installed)
from .modules.commons import recursive_munch, build_model, load_checkpoint
from .hf_utils import load_custom_model_from_hf
from .modules.audio import mel_spectrogram

class Inference:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.model, self.speechtokenizer_set, self.f0_extractor, self.bigvgan_model, \
        self.campplus_model, self.to_mel, self.mel_fn_args = self.load_models()
        self.sr = self.mel_fn_args['sampling_rate']
        self.f0_condition = args.f0_condition
        self.auto_f0_adjust = args.auto_f0_adjust
        self.pitch_shift = args.semi_tone_shift

    def load_models(self):
        args = self.args
        device = self.device

        # Load DiT model and configuration
        if not args.f0_condition:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
            )
            f0_extractor = None
        else:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
                "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
            )
            # f0 extractor
            from .modules.rmvpe import RMVPE
            model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
            f0_extractor = RMVPE(model_path, is_half=False, device=device)

        config = yaml.safe_load(open(dit_config_path, 'r'))
        model_params = recursive_munch(config['model_params'])
        model = build_model(model_params, stage='DiT')
        hop_length = config['preprocess_params']['spect_params']['hop_length']
        sr = config['preprocess_params']['sr']

        # Load checkpoints
        model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path,
                                         load_only_params=True, ignore_modules=[], is_distributed=False)
        for key in model:
            model[key].eval()
            model[key].to(device)
        model['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Load additional modules
        from .modules.campplus.DTDNN import CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval()
        campplus_model.to(device)

        from .modules.bigvgan import bigvgan
        bigvgan_name = 'nvidia/bigvgan_v2_22khz_80band_256x' if sr == 22050 else 'nvidia/bigvgan_v2_44khz_128band_512x'
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)

        # Remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)

        # Load speech tokenizer
        if model_params.speech_tokenizer.type == "facodec":
            ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec", 'pytorch_model.bin', 'config.yml')

            codec_config = yaml.safe_load(open(config_path))
            codec_model_params = recursive_munch(codec_config['model_params'])
            codec_encoder = build_model(codec_model_params, stage="codec")

            ckpt_params = torch.load(ckpt_path, map_location="cpu")

            for key in codec_encoder:
                codec_encoder[key].load_state_dict(ckpt_params[key], strict=False)
            for key in codec_encoder:
                codec_encoder[key].eval()
                codec_encoder[key].to(device)
            speechtokenizer_set = ('facodec', codec_encoder, None)
        elif model_params.speech_tokenizer.type == "whisper":
            from transformers import AutoFeatureExtractor, WhisperModel
            whisper_name = model_params.speech_tokenizer.get('whisper_name', "whisper-large-v3")
            whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
            del whisper_model.decoder
            whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
            speechtokenizer_set = ('whisper', whisper_model, whisper_feature_extractor)
        else:
            raise ValueError(f"Unsupported speech tokenizer type: {model_params.speech_tokenizer.type}")

        # Generate mel spectrograms
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": sr,
            "fmin": config['preprocess_params'].get('fmin', 0),
            "fmax": None if config['preprocess_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }

        to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        return model, speechtokenizer_set, f0_extractor, bigvgan_model, campplus_model, to_mel, mel_fn_args

    def adjust_f0_semitones(self, f0_sequence, n_semitones):
        factor = 2 ** (n_semitones / 12)
        return f0_sequence * factor

    @torch.no_grad()
    def inference(self, source: bytes, target_path):
        args = self.args
        device = self.device
        sr = self.sr
        f0_condition = self.f0_condition
        auto_f0_adjust = self.auto_f0_adjust
        pitch_shift = self.pitch_shift

        # Load source and target audio
        source_audio = librosa.load(source, sr=sr)[0]
        ref_audio = librosa.load(target_path, sr=sr)[0]

        # Trim audio to 30 seconds
        source_audio = source_audio[:sr * 30]
        ref_audio = ref_audio[:sr * 30]

        source_audio_tensor = torch.tensor(source_audio).unsqueeze(0).float().to(device)
        ref_audio_tensor = torch.tensor(ref_audio).unsqueeze(0).float().to(device)

        # Resample audio for different models
        source_waves_16k = torchaudio.functional.resample(source_audio_tensor, sr, 16000)
        ref_waves_16k = torchaudio.functional.resample(ref_audio_tensor, sr, 16000)

        # Extract speech tokens
        S_alt, S_ori = self.extract_speech_tokens(source_audio_tensor, ref_audio_tensor, source_waves_16k, ref_waves_16k)

        # Generate mel spectrograms
        mel = self.to_mel(source_audio_tensor.to(device).float())
        mel2 = self.to_mel(ref_audio_tensor.to(device).float())

        # Calculate target lengths
        length_adjust = args.length_adjust
        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        # Extract style embeddings
        style2 = self.extract_style_embedding(ref_waves_16k)

        # Extract F0 features if needed
        if f0_condition:
            shifted_f0_alt, F0_ori = self.extract_f0_features(source_waves_16k, ref_waves_16k)
        else:
            shifted_f0_alt, F0_ori = None, None

        # Length regulation
        cond = self.length_regulation(S_alt, target_lengths, shifted_f0_alt)
        prompt_condition = self.length_regulation(S_ori, target2_lengths, F0_ori)
        cat_condition = torch.cat([prompt_condition, cond], dim=1)

        # Voice conversion
        vc_target = self.voice_conversion(cat_condition, mel2, style2)

        # Convert to waveform
        vc_wave = self.bigvgan_model(vc_target).squeeze(1)

        # Return the audio binary data
        audio_binary: ndarray[float32]  = vc_wave.cpu().numpy()
        return audio_binary

    def extract_speech_tokens(self, source_audio, ref_audio, source_waves_16k, ref_waves_16k):
        device = self.device
        speechtokenizer_set = self.speechtokenizer_set

        if speechtokenizer_set[0] == 'facodec':
            codec_encoder = speechtokenizer_set[1]

            # Process source audio
            converted_waves_24k = torchaudio.functional.resample(source_audio, self.sr, 24000)
            waves_input = converted_waves_24k.unsqueeze(1)
            z = codec_encoder['encoder'](waves_input)
            _, codes = codec_encoder['quantizer'](z, waves_input)
            S_alt = torch.cat([codes[1], codes[0]], dim=1)

            # Process reference audio
            waves_24k = torchaudio.functional.resample(ref_audio, self.sr, 24000)
            waves_input = waves_24k.unsqueeze(1)
            z = codec_encoder['encoder'](waves_input)
            _, codes = codec_encoder['quantizer'](z, waves_input)
            S_ori = torch.cat([codes[1], codes[0]], dim=1)

        elif speechtokenizer_set[0] == 'whisper':
            whisper_model = speechtokenizer_set[1]
            whisper_feature_extractor = speechtokenizer_set[2]

            # Process source audio
            converted_waves_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)
            alt_inputs = whisper_feature_extractor([converted_waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            alt_input_features = whisper_model._mask_input_features(
                alt_inputs.input_features, attention_mask=alt_inputs.attention_mask).to(device)
            alt_outputs = whisper_model.encoder(
                alt_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            S_alt = alt_outputs.last_hidden_state.to(torch.float32)
            S_alt = S_alt[:, :converted_waves_16k.size(-1) // 320 + 1]

            # Process reference audio
            ori_inputs = whisper_feature_extractor([ref_waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            ori_outputs = whisper_model.encoder(
                ori_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :ref_waves_16k.size(-1) // 320 + 1]
        else:
            raise ValueError(f"Unsupported speech tokenizer type: {speechtokenizer_set[0]}")

        return S_alt, S_ori

    def extract_style_embedding(self, ref_waves_16k):
        device = self.device
        campplus_model = self.campplus_model

        feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                                  num_mel_bins=80,
                                                  dither=0,
                                                  sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))
        return style2

    def extract_f0_features(self, source_waves_16k, ref_waves_16k):
        device = self.device
        f0_extractor = self.f0_extractor
        auto_f0_adjust = self.auto_f0_adjust
        pitch_shift = self.pitch_shift

        F0_ori = f0_extractor.infer_from_audio(ref_waves_16k[0], thred=0.03)
        F0_alt = f0_extractor.infer_from_audio(source_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)
        # Shift alt log f0 level to ori log f0 level
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = self.adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)

        return shifted_f0_alt, F0_ori

    def length_regulation(self, S, target_lengths, f0):
        model = self.model
        cond, _, _, _, _ = model['length_regulator'](S, ylens=target_lengths, n_quantizers=3, f0=f0)
        return cond

    def voice_conversion(self, cat_condition, mel2, style2):
        args = self.args
        model = self.model
        diffusion_steps = args.diffusion_steps
        inference_cfg_rate = args.inference_cfg_rate

        vc_target = model['cfm'].inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2, style2, None, diffusion_steps,
            inference_cfg_rate=inference_cfg_rate
        )
        vc_target = vc_target[:, :, mel2.size(-1):]
        return vc_target
    

if __name__ == "__main__":
    # Define arguments (you can adjust these as needed)
    class Args:
        f0_condition = True
        auto_f0_adjust = True
        semi_tone_shift = 0
        length_adjust = 1.0
        diffusion_steps = 100
        inference_cfg_rate = 1.0

    args = Args()

    # Initialize the Inference class
    inference_engine = Inference(args)

    # Specify the paths to the source and target audio files
    source_audio_path = './testWavs/001-sibutomo.wav'  # Replace with the actual path
    target_audio_path = './testWavs/nc357342.wav'  # Replace with the actual path

    # Perform inference
    converted_audio_binary = inference_engine.inference(source_audio_path, target_audio_path)

    # Save the converted audio to a file for verification
    output_path = 'converted_audio.wav'  # Specify the desired output path
    import soundfile as sf
    sf.write(output_path, converted_audio_binary.T, inference_engine.sr)

    print(f"Converted audio saved to {output_path}")
