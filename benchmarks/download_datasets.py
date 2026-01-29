"""Automatic dataset downloader for CLEVR, VQA v2.0, and GQA.

Downloads and prepares benchmark datasets with progress tracking.
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse
import hashlib
import json


DATASET_INFO = {
    "clevr": {
        "name": "CLEVR v1.0",
        "url": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
        "size_gb": 18.0,
        "md5": None,  # Optional verification
        "extract_dir": "CLEVR_v1.0",
    },
    "clevr_mini": {
        "name": "CLEVR v1.0 (Mini - 10k samples)",
        "url": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
        "size_gb": 1.5,
        "extract_dir": "CLEVR_v1.0",
        "subset": True,
    },
    "vqa": {
        "name": "VQA v2.0",
        "urls": {
            "train_images": "http://images.cocodataset.org/zips/train2014.zip",
            "val_images": "http://images.cocodataset.org/zips/val2014.zip",
            "train_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
            "val_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
            "train_annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
            "val_annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        },
        "size_gb": 25.0,
        "extract_dir": "VQA",
    },
    "gqa": {
        "name": "GQA",
        "urls": {
            "images": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
            "questions": "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
            "scene_graphs": "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip",
        },
        "size_gb": 20.0,
        "extract_dir": "GQA",
    },
}


def download_file(url, dest_path, desc=None):
    """Download file with progress bar."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    if dest_path.exists():
        print(f"✓ {dest_path.name} already exists, skipping download")
        return
    
    print(f"Downloading {desc or dest_path.name}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=desc or dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    print(f"✓ Downloaded to {dest_path}")


def extract_archive(archive_path, extract_to, desc=None):
    """Extract zip or tar archive."""
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    
    print(f"Extracting {desc or archive_path.name}...")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in tqdm(members, desc=f"Extracting"):
                zip_ref.extract(member, extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc=f"Extracting"):
                tar_ref.extract(member, extract_to)
    
    print(f"✓ Extracted to {extract_to}")


def download_clevr(data_root, mini=False):
    """Download CLEVR dataset."""
    print("\n" + "="*70)
    print("Downloading CLEVR v1.0" + (" (Mini)" if mini else ""))
    print("="*70)
    
    dataset_key = "clevr_mini" if mini else "clevr"
    info = DATASET_INFO[dataset_key]
    
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Download
    zip_path = data_root / "CLEVR_v1.0.zip"
    download_file(info["url"], zip_path, desc="CLEVR v1.0")
    
    # Extract
    extract_archive(zip_path, data_root, desc="CLEVR v1.0")
    
    # Create mini subset if requested
    if mini:
        create_clevr_mini(data_root / info["extract_dir"])
    
    print(f"\n✓ CLEVR ready at: {data_root / info['extract_dir']}")
    return data_root / info["extract_dir"]


def create_clevr_mini(clevr_root):
    """Create mini CLEVR subset (10k train, 1k val)."""
    import shutil
    
    clevr_root = Path(clevr_root)
    mini_root = clevr_root.parent / "CLEVR_mini"
    
    if mini_root.exists():
        print("✓ CLEVR mini already exists")
        return
    
    print("\nCreating CLEVR mini subset...")
    mini_root.mkdir(exist_ok=True)
    
    # Copy structure
    for split in ['train', 'val']:
        n_samples = 10000 if split == 'train' else 1000
        
        # Images
        src_images = clevr_root / "images" / split
        dst_images = mini_root / "images" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        
        if src_images.exists():
            all_images = sorted(src_images.glob("*.png"))[:n_samples]
            for img in tqdm(all_images, desc=f"Copying {split} images"):
                shutil.copy(img, dst_images / img.name)
        
        # Questions
        src_questions = clevr_root / "questions" / f"CLEVR_{split}_questions.json"
        if src_questions.exists():
            with open(src_questions) as f:
                data = json.load(f)
            
            # Subset questions
            data['questions'] = data['questions'][:n_samples]
            
            dst_questions = mini_root / "questions" / f"CLEVR_{split}_questions.json"
            dst_questions.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dst_questions, 'w') as f:
                json.dump(data, f)
    
    print(f"✓ Created CLEVR mini at {mini_root}")


def download_vqa(data_root):
    """Download VQA v2.0 dataset."""
    print("\n" + "="*70)
    print("Downloading VQA v2.0")
    print("="*70)
    print("\n⚠️  Warning: VQA v2.0 is ~25GB. This may take a while.\n")
    
    info = DATASET_INFO["vqa"]
    data_root = Path(data_root) / info["extract_dir"]
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Download all components
    for component, url in info["urls"].items():
        filename = url.split("/")[-1]
        dest = data_root / filename
        
        download_file(url, dest, desc=f"VQA {component}")
        extract_archive(dest, data_root, desc=f"VQA {component}")
    
    print(f"\n✓ VQA v2.0 ready at: {data_root}")
    return data_root


def download_gqa(data_root):
    """Download GQA dataset."""
    print("\n" + "="*70)
    print("Downloading GQA")
    print("="*70)
    print("\n⚠️  Warning: GQA is ~20GB. This may take a while.\n")
    
    info = DATASET_INFO["gqa"]
    data_root = Path(data_root) / info["extract_dir"]
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Download all components
    for component, url in info["urls"].items():
        filename = url.split("/")[-1]
        dest = data_root / filename
        
        download_file(url, dest, desc=f"GQA {component}")
        extract_archive(dest, data_root, desc=f"GQA {component}")
    
    print(f"\n✓ GQA ready at: {data_root}")
    return data_root


def check_disk_space(required_gb):
    """Check available disk space."""
    import shutil
    
    stat = shutil.disk_usage(".")
    available_gb = stat.free / (1024**3)
    
    if available_gb < required_gb:
        print(f"\n⚠️  Warning: Only {available_gb:.1f}GB available, {required_gb:.1f}GB required")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Download cancelled")
            sys.exit(0)
    else:
        print(f"✓ Sufficient disk space: {available_gb:.1f}GB available")


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="clevr_mini",
        choices=["clevr", "clevr_mini", "vqa", "gqa", "all"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--skip-space-check",
        action="store_true",
        help="Skip disk space check"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("NEUROSYMBOLIC-T4 DATASET DOWNLOADER")
    print("="*70)
    
    # Calculate required space
    if args.dataset == "all":
        required_space = sum(info["size_gb"] for info in DATASET_INFO.values())
    elif args.dataset in DATASET_INFO:
        required_space = DATASET_INFO[args.dataset]["size_gb"]
    else:
        required_space = 0
    
    print(f"\nEstimated download size: {required_space:.1f}GB")
    
    # Check disk space
    if not args.skip_space_check and required_space > 0:
        check_disk_space(required_space * 1.5)  # 1.5x for extraction
    
    # Download
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == "clevr" or args.dataset == "all":
        download_clevr(data_root, mini=False)
    
    if args.dataset == "clevr_mini":
        download_clevr(data_root, mini=True)
    
    if args.dataset == "vqa" or args.dataset == "all":
        download_vqa(data_root)
    
    if args.dataset == "gqa" or args.dataset == "all":
        download_gqa(data_root)
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"\nDatasets saved to: {data_root.absolute()}")
    print("\nNext steps:")
    print("  1. Run training: python train_benchmarks.py --dataset clevr")
    print("  2. Or use Colab: https://colab.research.google.com/github/...")
    print("="*70)


if __name__ == "__main__":
    main()