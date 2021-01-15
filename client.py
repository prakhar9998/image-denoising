import requests
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

url = "http://60479b327b3c.ngrok.io/denoise"


def plot_results_side_by_side(orig, res_imgs):
    # orig is an image
    # res is a list of tuples (image, title)

    total = len(res_imgs)
    fig, axarrs = plt.subplots(1, total + 1)

    axarrs[0].imshow(orig)
    axarrs[0].set(title="Original")

    for i in range(1, total + 1):
        axarrs[i].imshow(res_imgs[i - 1][0])
        axarrs[i].set(title=res_imgs[i - 1][1])

    plt.show()


def check_response(res, parser):
    # checks server resposne and raises parser error
    if (res.status_code != 200):
        parser.error(
            f"Server responded with error code {res.status_code}. Message: {res.reason}")


def run():
    parser = argparse.ArgumentParser(description='Denoise image(s).')
    parser.add_argument("-i", dest="filename", required=True, metavar="IMAGE_FILE",
                        help="input an image file")
    parser.add_argument('--comp', dest="compare_methods", action="store_true",
                        help="to compare different methods side by side")

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        parser.error("The file %s does not exist!" % args.filename)

    input_img = Image.open(args.filename)
    input_img = input_img.convert("RGB")

    print("Waiting for server response...")

    if args.compare_methods:
        r_cnn = requests.post(url, data={'method': 'CNN'}, files={
                              'image': open(args.filename, 'rb')})
        r_nlm = requests.post(url, data={'method': 'NLM'}, files={
                              'image': open(args.filename, 'rb')})

        check_response(r_cnn, parser)
        check_response(r_nlm, parser)

        print("Denoising done!")

        denoised_cnn = Image.open(io.BytesIO(r_cnn.content))
        denoised_nlm = Image.open(io.BytesIO(r_nlm.content))

        plot_results_side_by_side(input_img, [
            (denoised_cnn, "Denoised (CNN)"), (denoised_nlm, "Denoised (NLM)")])
    else:
        r = requests.post(url, data={'method': 'CNN'}, files={
                          'image': open(args.filename, 'rb')})
        check_response(r, parser)
        print("Denoising done!")

        denoised_img = Image.open(io.BytesIO(r.content))

        plot_results_side_by_side(input_img, [(denoised_img, "Denoised")])


if __name__ == "__main__":
    run()
