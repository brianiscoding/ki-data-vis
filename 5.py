#!/usr/bin/env python
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from csv import reader
from pandas import read_csv
import numpy
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile

scale = 1 / 3
stack_length = 99
path_in = "input/"
path_temp = "temp/"
path_out = "output/"
f_names = [4, 6, 9, 11, 14, 45]


def create_stack(f):
    df = read_csv(f)

    z_max = df["normalized_block_sum"].max()
    x_min = df.x.min()
    x_max = df.x.max()
    y_min = df.y.min()
    y_max = df.y.max()
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    stack = [numpy.full(shape=(width, height), fill_value=255, dtype="uint8")]
    for _ in range(stack_length):
        stack.append(numpy.zeros(shape=(width, height), dtype="uint8"))

    with open(f, "r") as f_reader:
        f_reader = reader(f_reader)
        next(f_reader)
        for line in f_reader:
            z = round(float(line[5]) / z_max * stack_length)
            if z == 0:
                continue
            x = int(line[2]) - x_min
            y = int(line[3]) - y_min

            for i in range(z):
                stack[i + 1][x, y] = 255

    shape = (round(scale * width), round(scale * height))
    progress = tqdm(total=stack_length, desc=f"{f}")
    with ThreadPoolExecutor() as executor:
        pool = []
        for i, layer in enumerate(stack):
            pool.append(executor.submit(save_image, i, layer, shape))

        for _ in as_completed(pool):
            progress.update(1)
    progress.close()


def save_image(i, array, shape):
    Image.fromarray(array).resize(shape).save(f"{path_temp}{str(i).rjust(2, '0')}.png")


def compress(z_name):
    compression = zipfile.ZIP_DEFLATED
    zf = zipfile.ZipFile(z_name, mode="w")

    try:
        for i in range(stack_length + 1):
            zf.write(
                f"{path_temp}{str(i).rjust(2, '0')}.png",
                f"{str(i).rjust(2, '0')}.png",
                compress_type=compression,
            )
    except FileNotFoundError:
        print("An error occurred")
    finally:
        zf.close()


if __name__ == "__main__":
    for f_name in f_names:
        create_stack(f"{path_in}{f_name}.csv")
        compress(f"{path_out}{f_name}.zip")
