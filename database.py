import aiosqlite
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "road_data.db")

async def get_db():
    db = await aiosqlite.connect(DB_PATH, timeout=30)  # wait 30s before failing
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")   # allows concurrent reads+writes
    await db.execute("PRAGMA busy_timeout=30000") # 30s busy timeout
    try:
        yield db
    finally:
        await db.close()

async def init_db():
    async with aiosqlite.connect(DB_PATH, timeout=30) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                started_at TEXT DEFAULT (datetime('now')),
                ended_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trip_id INTEGER NOT NULL,
                timestamp TEXT DEFAULT (datetime('now')),
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                condition TEXT NOT NULL,
                speed REAL,
                accel_x REAL,
                accel_y REAL,
                accel_z REAL,
                accel_magnitude REAL,
                gyro_x REAL,
                gyro_y REAL,
                gyro_z REAL,
                gyro_magnitude REAL,
                accuracy REAL,
                altitude REAL,
                heading REAL,
                FOREIGN KEY (trip_id) REFERENCES trips(id)
            )
        """)
        await db.commit()