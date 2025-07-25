import sys
sys.path.append("AudioLDM-with-LoRA")
import os
import pandas as pd
import yaml
import script.utilities.audio as Audio
from script.utilities.tools import load_json
from .dataset_plugin import *
from librosa.filters import mel as librosa_mel_fn
import random
from transformers import RobertaTokenizerFast
from torch.utils.data import Dataset
from datasets import DatasetDict
import torch.nn.functional
import torch
import numpy as np
import torchaudio
import json
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

class HfAudioDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        split="train",
        waveform_only=False,
        add_ons=[],
    ):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("cvssp/audioldm-s-full-v2", subfolder="tokenizer")
        self.split = split
        self.pad_wav_start_sample = 0
        self.trim_wav = False
        self.waveform_only = waveform_only
        self.add_ons = [eval(x) for x in add_ons]

        self.build_setting_parameters()
        self.id2label, self.index_dict, self.num2label = {}, {}, {}

        # For an external dataset
        self.dataset_name = dataset_name
        self.build_dataset()

        self.build_dsp()
        self.label_num = len(self.index_dict)
        print("Dataset initialize finished")

    def build_setting_parameters(self):
        self.melbins = 64  # preprocessing.mel.n_mel_channels
        self.sampling_rate = 16000  # preprocessing.audio.sampling_rate
        self.hopsize = 160  # preprocessing.stft.hop_length
        self.duration = 10.24  # preprocessing.audio.duration
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize) 

        self.mixup = 0.0
        if "train" not in self.split:
            self.mixup = 0.0

    def build_dsp(self):
        self.mel_basis = {}
        self.hann_window = {}

        self.filter_length = 1024  # preprocessing.stft.filter_length
        self.hop_length = 160      # preprocessing.stft.hop_length
        self.win_length = 1024     # preprocessing.stft.win_length
        self.n_mel = 64            # preprocessing.mel.n_mel_channels
        self.sampling_rate = 16000  # preprocessing.audio.sampling_rate
        self.mel_fmin = 0          # preprocessing.mel.mel_fmin
        self.mel_fmax = 8000       # preprocessing.mel.mel_fmax

        self.STFT = Audio.stft.TacotronSTFT(
            1024,   # filter_length
            160,    # hop_length
            1024,   # win_length
            64,     # n_mel_channels
            16000,  # sampling_rate
            0,      # mel_fmin
            8000    # mel_fmax
        )
    def build_dataset(self):
        
        self.data = []
        
        if hasattr(self.dataset_name, 'features'):  # HuggingFace Dataset의 특징
            print("Detected HuggingFace dataset. Extracting relevant fields...")
            hf_dataset = self.dataset_name

            for item in hf_dataset:
                self.data.append({
                    "wav": item["audio"]["array"],  # 로컬 경로가 없으면 None
                    "sr": item["audio"]["sampling_rate"],
                    "caption": item.get("caption", ""),
                    "labels": item.get("labels", ""),       # 없으면 빈 문자열
                    "seg_label": item.get("seg_label", "")   # 없으면 빈 문자열
                })
        else:
            raise Exception("Invalid data format")
        
        print("Data size: {}".format(len(self.data)))

    def __getitem__(self, index):
        (
            fname,
            waveform,
            stft,
            log_mel_spec,
            label_vector,  # the one-hot representation of the audio class
            # the metadata of the sampled audio file and the mixup audio file (if exist)
            (datum, mix_datum),
            random_start,
        ) = self.feature_extraction(index)
        
        text = self.get_sample_text_caption(datum, mix_datum, label_vector)
        caption = text if text else "" # 리스트에서 첫 번째 캡션 추출 (또는 빈 문자열)

        # 텍스트 토큰화
        inputs = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        # input_ids = inputs.input_ids.squeeze(0)
        # attention_mask = inputs.attention_mask.squeeze(0)


        data = {
            "text": input_ids,  # list
            "attention_mask": attention_mask,
            # "fname": self.text_to_filename(text) if (not fname) else fname,  # list
            # tensor, [batchsize, class_num]
            "label_vector": label_vector.float() if isinstance(label_vector, torch.Tensor) else torch.zeros(512).float(),
            # tensor, [batchsize, 1, samples_num]
            "waveform": "" if (waveform is None) else waveform.float(),
            # tensor, [batchsize, t-steps, f-bins]
            "stft": "" if (stft is None) else stft.float(),
            # tensor, [batchsize, t-steps, mel-bins]
            "log_mel_spec": "" if (log_mel_spec is None) else log_mel_spec.float(),
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "random_start_sample_in_original_audio_file": random_start,
        }

        caption = data.get("text", None)
        if caption is None:
            print(f"Warning: No text found for index {index}")
            caption = ""
            


        return data

    def __len__(self):
        return len(self.data)

    def resample(self, waveform, sr):
        waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        return waveform

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

    def random_segment_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        for i in range(10):
            random_start = int(self.random_uniform(0, waveform_length - target_length))
            if torch.max(
                torch.abs(waveform[:, random_start : random_start + target_length])
                > 1e-4
            ):
                break

        return waveform[:, random_start : random_start + target_length], random_start

    def pad_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == target_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        if self.pad_wav_start_sample is None:
            rand_start = int(self.random_uniform(0, target_length - waveform_length))
        else:
            rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length
            while start - chunk_size > 0:
                if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                    start -= chunk_size
                else:
                    break
            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    def read_wav_file(self, filename):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        waveform, sr = torchaudio.load(filename)

        waveform, random_start = self.random_segment_wav(
            waveform, target_length=int(sr * self.duration)
        )

        waveform = self.resample(waveform, sr)
        # random_start = int(random_start * (self.sampling_rate / sr))

        waveform = waveform.numpy()[0, ...]

        waveform = self.normalize_wav(waveform)

        if self.trim_wav:
            waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]
        waveform = self.pad_wav(
            waveform, target_length=int(self.sampling_rate * self.duration)
        )
        return waveform, random_start

    def read_audio_file(self, filename, filename2=None):
        if os.path.exists(filename):
            waveform, random_start = self.read_wav_file(filename)
        else:
            print(
                'Non-fatal Warning [dataset.py]: The wav path "',
                filename,
                '" is not find in the metadata. Use empty waveform instead. This is normal in the inference process.',
            )
            target_length = int(self.sampling_rate * self.duration)
            waveform = torch.zeros((1, target_length))
            random_start = 0

        # log_mel_spec, stft = self.wav_feature_extraction_torchaudio(waveform) # this line is faster, but this implementation is not aligned with HiFi-GAN
        if not self.waveform_only:
            log_mel_spec, stft = self.wav_feature_extraction(waveform)
        else:
            # Load waveform data only
            # Use zero array to keep the format unified
            log_mel_spec, stft = None, None

        return log_mel_spec, stft, waveform, random_start

    def get_sample_text_caption(self, datum, mix_datum, label_indices):
        text = self.label_indices_to_text(datum, label_indices)
        if mix_datum is not None:
            text += " " + self.label_indices_to_text(mix_datum, label_indices)
        return text

    def mel_spectrogram_train(self, y):
        if torch.min(y) < -1.0:
            print("train min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("train max value is ", torch.max(y))

        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                self.sampling_rate,
                self.filter_length,
                self.n_mel,
                self.mel_fmin,
                self.mel_fmax,
            )
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(
                y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )

        y = y.squeeze(1)

        stft_spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(y.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        stft_spec = torch.abs(stft_spec)

        mel = spectral_normalize_torch(
            torch.matmul(
                self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec
            )
        )

        return mel[0], stft_spec[0]

    # This one is significantly slower than "wav_feature_extraction_torchaudio" if num_worker > 1
    def wav_feature_extraction(self, waveform):
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        # log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)[0]
        log_mel_spec, stft = self.mel_spectrogram_train(waveform.unsqueeze(0))

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        return log_mel_spec, stft

    # @profile
    # def wav_feature_extraction_torchaudio(self, waveform):
    #     waveform = waveform[0, ...]
    #     waveform = torch.FloatTensor(waveform)

    #     stft = self.stft_transform(waveform)
    #     mel_spec = self.melscale_transform(stft)
    #     log_mel_spec = torch.log(mel_spec + 1e-7)

    #     log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    #     stft = torch.FloatTensor(stft.T)

    #     log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
    #     return log_mel_spec, stft

    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec

    def _read_datum_caption(self, datum):
        caption_keys = [x for x in datum.keys() if ("caption" in x)]
        random_index = torch.randint(0, len(caption_keys), (1,))[0].item()
        return datum[caption_keys[random_index]]

    def _is_contain_caption(self, datum):
        caption_keys = [x for x in datum.keys() if ("caption" in x)]
        return len(caption_keys) > 0

    def label_indices_to_text(self, datum, label_indices):
        if self._is_contain_caption(datum):
            return self._read_datum_caption(datum)
        elif "caption" in datum.keys():
            name_indices = torch.where(label_indices > 0.1)[0]
            # description_header = "This audio contains the sound of "
            description_header = ""
            labels = ""
            for id, each in enumerate(name_indices):
                if id == len(name_indices) - 1:
                    labels += "%s." % self.num2label[int(each)]
                else:
                    labels += "%s, " % self.num2label[int(each)]
            return description_header + labels
        else:
            return ""  # TODO, if both label and caption are not provided, return empty string

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def frequency_masking(self, log_mel_spec, freqm):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq - mask_len))
        log_mel_spec[:, mask_start : mask_start + mask_len, :] *= 0.0
        return log_mel_spec

    def time_masking(self, log_mel_spec, timem):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps - mask_len))
        log_mel_spec[:, :, mask_start : mask_start + mask_len] *= 0.0
        return log_mel_spec

    def feature_extraction(self, index):
        if index > len(self.data) - 1:
            print(
                "The index of the dataloader is out of range: %s/%s"
                % (index, len(self.data))
            )
            index = random.randint(0, len(self.data) - 1)

        # Read wave file and extract feature
        while True:
            try:
                label_indices = np.zeros(self.label_num, dtype=np.float32)
                datum = self.data[index]

                waveform, random_start = self.read_hf_audio(datum["wav"], datum["sr"])

                log_mel_spec, stft = self.wav_feature_extraction(waveform)
                
                mix_datum = None

                if self.label_num > 0 and "labels" in datum.keys():
                    for label_str in datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] = 1.0

                # If the key "label" is not in the metadata, return all zero vector
                label_indices = torch.FloatTensor(label_indices)
                break
            except Exception as e:
                print("❌ Error during audio feature extraction:")
                print("   → Error message:", e)
                print("   → datum keys:", datum.keys())
                print("   → index:", index)
                raise e  # 바로 예외를 다시 던져서 중단시킴

        # The filename of the wav file
        fname = datum["wav"]
        # t_step = log_mel_spec.size(0)
        # waveform = torch.FloatTensor(waveform[..., : int(self.hopsize * t_step)])
        waveform = torch.FloatTensor(waveform)

        return (
            fname,
            waveform,
            stft,
            log_mel_spec,
            label_indices,
            (datum, mix_datum),
            random_start,
        )
    
    def read_hf_audio(self, array, sr):
        # (1) waveform shape: (1, T)
        waveform = array[None, :]

        # (2) 랜덤 segment
        waveform, random_start = self.random_segment_wav(
            waveform, target_length=int(sr * self.duration)
        )

        # (3) torch tensor로 변환 (resample 전에!)
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        # (4) resample
        waveform = self.resample(waveform, sr)

        # (5) numpy → normalize
        waveform = waveform.numpy()[0, ...]
        waveform = self.normalize_wav(waveform)

        # (6) trim
        if self.trim_wav:
            waveform = self.trim_wav(waveform)

        # (7) padding
        waveform = waveform[None, ...]
        waveform = self.pad_wav(waveform, target_length=int(self.sampling_rate * self.duration))

        return waveform, random_start