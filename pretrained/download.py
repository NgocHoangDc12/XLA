import os
import zipfile
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract(url, extract_to="."):
    filename = os.path.basename(url)
    zip_path = os.path.join(extract_to, filename)
    print(f"Downloading {url} ...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)

    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    if os.path.exists(zip_path):
        os.remove(zip_path)
    print(f"Finished downloading and extracting {filename}!\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Download pre-trained weights for CO-SPY")
    parser.add_argument("--model", type=str, default="sd-v1_4", choices=["sd-v1_4", "progan", "all"],
                        help="Which pre-trained weights to download: sd-v1_4, progan, or all")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    urls = {
        "sd-v1_4": "https://huggingface.co/ruojiruoli/Co-Spy-Pretrained-Weights/resolve/main/sd-v1_4.zip",
        "progan": "https://huggingface.co/ruojiruoli/Co-Spy-Pretrained-Weights/resolve/main/progan.zip",
    }

    if args.model in ["sd-v1_4", "all"]:
        download_and_extract(urls["sd-v1_4"], extract_to=script_dir)
    if args.model in ["progan", "all"]:
        download_and_extract(urls["progan"], extract_to=script_dir)

    # Clean __MACOSX if created
    macosx_dir = os.path.join(script_dir, "__MACOSX")
    if os.path.exists(macosx_dir):
        import shutil
        shutil.rmtree(macosx_dir)
