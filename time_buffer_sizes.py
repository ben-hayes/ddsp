import os
import pickle
import time

import click
import gin
import numpy as np
import pandas as pd
from tqdm import trange

import ddsp.training

BUFFER_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

@click.command()
@click.option("--dummy-data", prompt="dummy-data")
@click.option("--output-file", prompt="output file")
# @click.option("--tfrecord-pattern", prompt="tfrecord pattern")
@click.option("--model-dir", prompt="Model save dir")
@click.option("--num-iters", default=100)
@click.option("--batch-size", default=1)
@click.option("--model-name", default="ddsp")
@click.option("--device", default="cpu")
def main(dummy_data, output_file, model_dir, num_iters, batch_size, model_name, device):
    # data_provider = ddsp.training.data.TFRecordProvider(tfrecord_pattern)
    # dataset = data_provider.get_batch(batch_size=batch_size, shuffle=False, repeats=-1)
    # dummy_input = next(iter(dataset))
    with open(dummy_data, "rb") as f:
        dummy_input = pickle.load(f)

    gin.parse_config_file(os.path.join(model_dir, "operative_config-0.gin"))

    times = []
    for bs in BUFFER_SIZES:
        n_samples = bs
        time_steps = bs // 64
        gin_params = [
            'Harmonic.n_samples = {}'.format(n_samples),
            'FilteredNoise.n_samples = {}'.format(n_samples),
            'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
            'oscillator_bank.use_angular_cumsum = True',
        ]

        with gin.unlock_config():
            gin.parse_config(gin_params)

        model = ddsp.training.models.Autoencoder()

        b_data = dummy_input.copy()
        for k in b_data:
            if k == "audio":
                b_data[k] = b_data[k][..., :n_samples]
            else:
                b_data[k] = b_data[k][..., :time_steps]

        for i in trange(num_iters):
            start_time = time.time()
            outputs = model(b_data)
            audio_gen = model.get_audio_from_outputs(outputs)
            time_elapsed = time.time() - start_time
            times.append(
                [model_name, device if device == "cpu" else "gpu", bs, time_elapsed]
            )

    df = pd.DataFrame(times)
    df.to_csv(output_file)


if __name__ == "__main__":
    main()