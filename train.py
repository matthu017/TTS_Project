import pdb
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import vocoder.hparams as hp
from vocoder.display import stream, simple_table
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.gen_wavernn import gen_testset
from vocoder.models.fatchord_version import WaveRNN
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
from fairseq.models.wav2vec import Wav2VecModel
import os
from functools import partial
from geomloss import SamplesLoss

def infer_waveform(mel, normalize=True,  batched=True, target=8000, overlap=800,
                   progress_callback=None):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")

    if normalize:
        mel = mel / hp.mel_max_abs_value
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(mel, batched, target, overlap, hp.mu_law, progress_callback)
    return wav

class PerceptualLoss(nn.Module):
    def __init__(self, model_type='wav2vec', PRETRAINED_MODEL_PATH = '/fs/scratch/PAS2400/TTS_baseline_cpy/Real-Time-Voice-Cloning/vocoder/PFPL/wav2vec_large.pt'):
        super().__init__()
        self.model_type = model_type
        self.wass_dist = SamplesLoss()
        if model_type == 'wav2vec':
            #ckpt = torch.load(PRETRAINED_MODEL_PATH, map_location=torch.device('cpu'))
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([PRETRAINED_MODEL_PATH])
            #self.model = Wav2VecModel.build_model(ckpt['args'], task=None)
            #self.model.load_state_dict(ckpt['model'])
            self.model = model[0]
            self.model = self.model.feature_extractor
            self.model.eval()
        else:
            print('Please assign a loss model')
            sys.exit()

    def forward(self, y_hat, y):
        y_hat, y = map(self.model, [y_hat, y])
        return self.wass_dist(y_hat, y)
        # for PFPL-W-MAE or PFPL-W
        # return torch.abs(y_hat - y).mean()

def train(run_id: str, syn_dir: Path, voc_dir: Path, models_dir: Path, ground_truth: bool, save_every: int,
          backup_every: int, force_restart: bool):
    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    # Instantiate the model
    print("Initializing the model...")
    model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    
    for p in optimizer.param_groups:
        p["lr"] = hp.voc_lr
    loss_func = F.cross_entropy if model.mode == "RAW" else discretized_mix_logistic_loss
    per_loss = PerceptualLoss(model_type='wav2vec')
    
    if torch.cuda.is_available():
        per_loss = per_loss.cuda()
    
    #per_loss = per_loss.to(device)
    
    criterion = lambda y_hat, y: per_loss(y.float().squeeze(),y_hat.mean(1).squeeze()).mean() + loss_func(y_hat, y)
    # Load the weights
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True)
    weights_fpath = model_dir / "vocoder.pt"
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of WaveRNN from scratch\n")
        model.save(weights_fpath, optimizer)
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("WaveRNN weights loaded from step %d" % model.step)

    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.txt") if ground_truth else \
        voc_dir.joinpath("synthesized.txt")
    mel_dir = syn_dir.joinpath("mels") if ground_truth else voc_dir.joinpath("mels_gta")
    wav_dir = syn_dir.joinpath("audio")
    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Begin the training
    simple_table([('Batch size', hp.voc_batch_size),
                  ('LR', hp.voc_lr),
                  ('Sequence Len', hp.voc_seq_len)])
    #gen_testset(model, test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
    #                hp.voc_target, hp.voc_overlap, model_dir)

    for epoch in range(1, 35):
        data_loader = DataLoader(dataset, hp.voc_batch_size, shuffle=True, num_workers=2, collate_fn=collate_vocoder)
        start = time.time()
        running_loss = 0.
        #print("######### Training at epoch:   ", epoch, " #########\n")
        for i, (x, y, m) in enumerate(data_loader, 1):
            if torch.cuda.is_available():
                x, m, y = x.cuda(), m.cuda(), y.cuda() 
            #pdb.set_trace() 
            ## Data aug reverse ##            
            xx = torch.flip(x,[1])
            yy = torch.flip(y,[1])
            x = torch.cat((x,xx),0)
            y = torch.cat((y,yy),0)
            mm = torch.flip(m,[1])
            m = torch.cat((m,mm),0)
            # Forward pass
            #pdb.set_trace()
            y_hat = model(x, m)
            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif model.mode == 'MOL':
                y = y.float()
            y = y.unsqueeze(-1)

            # Backward pass
            #pdb.set_trace()
            #gen = model.generate(m,batched=False, target=8000, overlap=800, mu_law=True, progress_callback=None)
            #print(gen.shape,y.shape)
            loss = criterion(y_hat, y)
            #pdb.set_trace()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            step = model.get_step()
            k = step // 1000

            #if backup_every != 0 and step % backup_every == 0 :
            model.checkpoint(model_dir, optimizer)

            #if save_every != 0 and step % save_every == 0 :
            model.save(weights_fpath, optimizer)

            msg = f"| Epoch: {epoch} ({i}/{len(data_loader)}) | " \
                f"Loss: {avg_loss:.4f} | {speed:.1f} " \
                f"steps/s | Step: {k}k | "
            stream(msg)
        print("######### Training at epoch:   ", str(epoch), " Loss: "+str(avg_loss)+" #########\n")

        gen_testset(model, test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                    hp.voc_target, hp.voc_overlap, model_dir)
        print("")
