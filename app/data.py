from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import aiosqlite
from app.database import get_db
from app.ws_manager import manager

router = APIRouter()

ALERT_CONDITIONS = {"bad", "obstruction"}

class Reading(BaseModel):
    trip_id: int
    lat: float
    lng: float
    condition: str  # good, avg, bad, obstruction
    speed: Optional[float] = None
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    accel_magnitude: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None
    gyro_magnitude: Optional[float] = None
    accuracy: Optional[float] = None
    altitude: Optional[float] = None
    heading: Optional[float] = None

class TripCreate(BaseModel):
    name: Optional[str] = None

@router.post("/trips")
async def create_trip(body: TripCreate, db: aiosqlite.Connection = Depends(get_db)):
    name = body.name or f"Trip"
    cursor = await db.execute("INSERT INTO trips (name) VALUES (?)", (name,))
    await db.commit()
    return {"trip_id": cursor.lastrowid, "name": name}

@router.patch("/trips/{trip_id}/end")
async def end_trip(trip_id: int, db: aiosqlite.Connection = Depends(get_db)):
    await db.execute(
        "UPDATE trips SET ended_at = datetime('now') WHERE id = ?", (trip_id,)
    )
    await db.commit()
    return {"status": "ended"}

@router.post("/data")
async def post_reading(reading: Reading, db: aiosqlite.Connection = Depends(get_db)):
    if reading.condition not in ("good", "avg", "bad", "obstruction"):
        raise HTTPException(status_code=400, detail="Invalid condition")
    await db.execute("""
        INSERT INTO readings 
        (trip_id, lat, lng, condition, speed, accel_x, accel_y, accel_z, accel_magnitude,
         gyro_x, gyro_y, gyro_z, gyro_magnitude, accuracy, altitude, heading)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        reading.trip_id, reading.lat, reading.lng, reading.condition,
        reading.speed, reading.accel_x, reading.accel_y, reading.accel_z, reading.accel_magnitude,
        reading.gyro_x, reading.gyro_y, reading.gyro_z, reading.gyro_magnitude,
        reading.accuracy, reading.altitude, reading.heading
    ))
    await db.commit()

    # Broadcast alert to all connected clients for bad/obstruction readings
    if reading.condition in ALERT_CONDITIONS:
        await manager.broadcast({
            "type": "road_alert",
            "condition": reading.condition,
            "lat": reading.lat,
            "lng": reading.lng,
            "trip_id": reading.trip_id,
        })

    return {"status": "ok"}

@router.get("/segments")
async def get_segments(trip_id: Optional[int] = None, db: aiosqlite.Connection = Depends(get_db)):
    if trip_id:
        cursor = await db.execute(
            "SELECT * FROM readings WHERE trip_id = ? ORDER BY id ASC", (trip_id,)
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM readings ORDER BY trip_id, id ASC"
        )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]

@router.get("/trips")
async def get_trips(db: aiosqlite.Connection = Depends(get_db)):
    cursor = await db.execute("""
        SELECT t.id, t.name, t.started_at, t.ended_at,
               COUNT(r.id) as point_count
        FROM trips t
        LEFT JOIN readings r ON r.trip_id = t.id
        GROUP BY t.id
        ORDER BY t.id DESC
    """)
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]

@router.get("/trips/{trip_id}/stats")
async def get_trip_stats(trip_id: int, db: aiosqlite.Connection = Depends(get_db)):
    cursor = await db.execute("""
        SELECT condition, COUNT(*) as count,
               AVG(speed) as avg_speed,
               AVG(accel_magnitude) as avg_accel,
               AVG(gyro_magnitude) as avg_gyro
        FROM readings WHERE trip_id = ?
        GROUP BY condition
    """, (trip_id,))
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]
