"""Serve the review pages and MP4s with HTTP Range support for timestamp seeking."""
import argparse
from pathlib import Path
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
import uvicorn

root=Path(__file__).resolve().parents[1]/'review_page/dist'
app=Starlette(routes=[Mount('/',app=StaticFiles(directory=root,html=True,follow_symlink=True))])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8840)
    args=p.parse_args();uvicorn.run(app,host=args.host,port=args.port)
