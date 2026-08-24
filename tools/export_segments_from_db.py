#!/usr/bin/env python3
"""
Export readings from the local SQLite DB (app/road_data.db) to frontend/segments.json.

Usage:
  python3 tools/export_segments_from_db.py [--db app/road_data.db] [--out frontend/segments.json]
"""
import argparse, sqlite3, json, os

def export(db_path, out_path):
    if not os.path.exists(db_path):
        print('DB not found:', db_path)
        return 2
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute('SELECT * FROM readings ORDER BY trip_id, id ASC')
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    # Convert any bytes to str and ensure JSON serializable
    for r in rows:
        for k,v in list(r.items()):
            if isinstance(v, bytes): r[k]=v.decode('utf-8')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    print('Wrote', out_path, 'with', len(rows), 'rows')
    return 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='app/road_data.db')
    p.add_argument('--out', default='frontend/segments.json')
    args = p.parse_args()
    return export(args.db, args.out)

if __name__ == '__main__':
    raise SystemExit(main())
