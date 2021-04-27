import os
import pickle
import time

import click
import gin
import numpy as np
from scipy.stats import describe
from tqdm import trange

import ddsp.training


@click.command()
@click.option("--dummy-data", prompt="dummy-data")
@click.option("--model-dir", prompt="Model save dir")
@click.option("--num-iters", default=100)
@click.option("--batch-size", default=1)
def main(dummy_data, model_dir, num_iters, batch_size):
    # data_provider = ddsp.training.data.TFRecordProvider(tfrecord_pattern)
    # dataset = data_provider.get_batch(batch_size=batch_size, shuffle=False, repeats=-1)
    with open(dummy_data, "rb") as f:
        dummy_input = pickle.load(f)

    gin.parse_config_file(os.path.join(model_dir, "operative_config-0.gin"))
    model = ddsp.training.models.Autoencoder()
    model.restore(model_dir)

    # dummy_input = next(iter(dataset))

    times = []
    for i in trange(num_iters):
        start_time = time.time()
        model(dummy_input)
        time_elapsed = time.time() - start_time
        times.append(time_elapsed)

    print(describe(times))
    rtfs = np.array(times) / 4
    print("Mean RTF: %.4f" % np.mean(rtfs))
    print("90th percentile RTF: %.4f" % np.percentile(rtfs, 90))


if __name__ == "__main__":
    main()