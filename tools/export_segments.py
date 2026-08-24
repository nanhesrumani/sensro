#!/usr/bin/env python3
"""
Fetch /api/segments from a backend and write frontend/segments.json for GitHub Pages.

Usage:
  python tools/export_segments.py --backend http://localhost:8000
"""
import argparse, json
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

def fetch_segments(backend_url):
    url = backend_url.rstrip('/') + '/api/segments'
    req = Request(url, headers={'User-Agent': 'sensro-export/1.0'})
    with urlopen(req, timeout=15) as r:
        data = r.read()
        return json.loads(data)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', default='http://localhost:8000', help='Backend base URL')
    p.add_argument('--out', default='frontend/segments.json', help='Output file')
    args = p.parse_args()
    try:
        segments = fetch_segments(args.backend)
    except HTTPError as e:
        print('HTTP error:', e.code, e.reason)
        return 2
    except URLError as e:
        print('URL error:', e.reason)
        return 2
    except Exception as e:
        print('Fetch failed:', e)
        return 2
    with open(args.out, 'w') as f:
        json.dump(segments, f, indent=2)
    print('Wrote', args.out, 'from', args.backend)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
