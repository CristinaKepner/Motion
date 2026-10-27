"""Publish the actual speech demo and embedded event trace alongside the library pages."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
trace=json.loads((root/'results/genea_demo/trace.json').read_text())
source=(root/'review_page/src/speech_demo.html').read_text()
(root/'review_page/dist/A3_speech_demo.html').write_text(source.replace('__DATA__',json.dumps(trace,ensure_ascii=False).replace('</','<\\/')))
link=root/'review_page/dist/demo'
if not link.exists():link.symlink_to('../../results/genea_demo',target_is_directory=True)
