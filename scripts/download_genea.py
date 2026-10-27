"""Extract a reproducible training subset directly from the official ZIP using HTTP ranges.

Each extracted member is checked against the ZIP CRC and size. No motion is
trimmed or relabelled. The original WAV/TSV and attribution are retained.
"""
import argparse
import concurrent.futures
import io
import json
from pathlib import Path
import struct
import time
import urllib.request
import zipfile
import zlib

URL = 'https://zenodo.org/records/8199133/files/genea2023_trn.zip?download=1'
SIZE = 8621407555


def fetch(start, end):
    for attempt in range(6):
        try:
            req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=180) as r:
                if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                    raise ValueError('server did not honor the exact requested range')
                data = r.read()
            if len(data) != end-start+1:
                raise ValueError('incomplete HTTP range')
            return data
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(60, 5 * 2**attempt))


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.tail_start = SIZE - 1024*1024
        self.tail = fetch(self.tail_start, SIZE-1)

    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else SIZE)
        return self.pos
    def read(self, size=-1):
        end = SIZE if size < 0 else min(SIZE, self.pos+size)
        data = self.tail[self.pos-self.tail_start:end-self.tail_start] if self.pos >= self.tail_start else fetch(self.pos, end-1)
        self.pos = end
        return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('external/datasets/genea2023'))
    ap.add_argument('--files', type=int, default=24)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteZip()) as archive:
        entries = archive.infolist()
    (args.output/'zip_index.json').write_text(json.dumps([
        {'name': e.filename, 'size': e.file_size, 'compressed': e.compress_size, 'offset': e.header_offset}
        for e in entries], indent=2))
    bvhs = sorted([e for e in entries if '/main-agent/bvh/' in '/'+e.filename and e.filename.endswith('.bvh')], key=lambda e:e.filename)
    if not bvhs:
        raise ValueError('official archive layout changed; inspect zip_index.json')
    # Evenly distributed over the ordered training set; selection is independent of scores.
    selected = [bvhs[round(i*(len(bvhs)-1)/max(args.files-1,1))] for i in range(min(args.files,len(bvhs)))]
    stems = {Path(e.filename).stem for e in selected}
    take = [e for e in entries if not e.is_dir() and (
        ('/main-agent/' in '/'+e.filename and Path(e.filename).stem in stems and Path(e.filename).suffix in ('.bvh','.wav','.tsv'))
        or Path(e.filename).name in ('metadata.csv','LICENSE.txt','README.txt'))]
    print('selected', len(selected), 'motion files;', len(take), 'members;', sum(e.compress_size for e in take)/1e6, 'MB compressed', flush=True)

    def extract(e):
        rel = Path(e.filename)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('unsafe archive member')
        dest = args.output/rel
        def valid(data): return len(data)==e.file_size and zlib.crc32(data)==e.CRC
        if dest.exists() and valid(dest.read_bytes()):
            return {'file': e.filename, 'crc32': f'{e.CRC:08x}', 'bytes': e.file_size}
        # Local header may have extra fields absent from the central directory.
        first = fetch(e.header_offset, min(SIZE-1, e.header_offset+4095))
        assert first[:4] == b'PK\x03\x04'
        name_len, extra_len = struct.unpack_from('<HH', first, 26)
        start = e.header_offset+30+name_len+extra_len
        compressed = fetch(start, start+e.compress_size-1) if e.compress_size else b''
        data = zlib.decompress(compressed,-15) if e.compress_type==zipfile.ZIP_DEFLATED else compressed
        if not valid(data): raise ValueError(f'CRC/size mismatch: {e.filename}')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
        print('verified',e.filename,e.file_size,flush=True)
        return {'file': e.filename, 'crc32': f'{e.CRC:08x}', 'bytes': e.file_size}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(extract,take))
    # Stable unified entry point; keep official separate bvh/wav/tsv folders too.
    unified=args.output/'trn'/'bvh';unified.mkdir(parents=True,exist_ok=True)
    import os
    for e in take:
        if Path(e.filename).stem in stems:
            link=unified/Path(e.filename).name
            if not link.exists(): link.symlink_to(os.path.relpath(args.output/e.filename,unified))
    (args.output/'subset_manifest.json').write_text(json.dumps({
        'source':URL,'record':'https://zenodo.org/records/8199133','license':'CC BY-NC 4.0',
        'selection':'24 evenly spaced main-agent training files, before evaluation',
        'archive_md5':'409a321605d0f5bbd042f943fed4bd3e','archive_md5_verified':False,
        'member_crc_verified':True,'members':manifest},indent=2))


if __name__=='__main__': main()
