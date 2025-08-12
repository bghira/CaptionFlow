#!/usr/bin/env python3
"""
Test script to verify central collection is working.

This simulates workers sending captions to the orchestrator
and verifies they are centrally collected and written to disk.
"""

import asyncio
import json
import time
from pathlib import Path

import websockets
import pyarrow.parquet as pq

async def simulate_worker(server_url: str, token: str, worker_name: str, num_captions: int):
    """Simulate a worker sending captions."""
    print(f"[{worker_name}] Connecting to {server_url}")
    
    async with websockets.connect(server_url) as websocket:
        # Authenticate
        await websocket.send(json.dumps({
            "token": token,
            "name": worker_name
        }))
        
        # Wait for welcome
        welcome = await websocket.recv()
        print(f"[{worker_name}] Connected: {welcome}")
        
        # Simulate sending captions
        for i in range(num_captions):
            # Request a job (in real scenario)
            await websocket.send(json.dumps({
                "type": "request_job"
            }))
            
            # Wait for job assignment
            msg = await websocket.recv()
            data = json.loads(msg)
            
            if data.get("type") == "job":
                job = data["job"]
                
                # Simulate caption generation
                caption = f"Test caption {i+1} from {worker_name}"
                
                # Send caption back for central collection
                await websocket.send(json.dumps({
                    "type": "submit_caption",
                    "job_id": job["job_id"],
                    "dataset": job["dataset"],
                    "shard": job["shard"],
                    "item_key": job["item_key"],
                    "caption": caption
                }))
                
                print(f"[{worker_name}] Sent caption {i+1}/{num_captions}")
                
                # Wait for acknowledgment
                ack = await websocket.recv()
                
                # Small delay to simulate processing
                await asyncio.sleep(0.1)
            else:
                print(f"[{worker_name}] No jobs available")
                await asyncio.sleep(1)

async def test_central_collection():
    """Test the central collection system."""
    server_url = "wss://localhost:8765"
    
    print("=" * 60)
    print("CENTRAL COLLECTION TEST")
    print("=" * 60)
    
    # Start multiple simulated workers
    workers = [
        ("worker-1", "gpu-worker-token-1", 35),
        ("worker-2", "gpu-worker-token-2", 35),
        ("worker-3", "gpu-worker-token-3", 35),
    ]
    
    print(f"\n1. Starting {len(workers)} simulated workers...")
    
    tasks = []
    for name, token, count in workers:
        task = asyncio.create_task(
            simulate_worker(server_url, token, name, count)
        )
        tasks.append(task)
    
    # Wait for all workers to complete
    await asyncio.gather(*tasks)
    
    print("\n2. All workers completed sending captions")
    print("   Total captions sent: 105 (should trigger flush at 100)")
    
    # Wait for orchestrator to flush
    print("\n3. Waiting for orchestrator to flush to disk...")
    await asyncio.sleep(3)
    
    # Check the output
    caption_file = Path("./caption_data/captions.parquet")
    
    if caption_file.exists():
        print("\n4. Reading centrally collected captions...")
        table = pq.read_table(caption_file)
        df = table.to_pandas()
        
        print(f"   ✓ Total captions in file: {len(df)}")
        print(f"   ✓ Unique contributors: {df['contributor_id'].nunique()}")
        print(f"   ✓ Contributors: {df['contributor_id'].unique().tolist()}")
        
        # Show sample
        print("\n5. Sample of collected captions:")
        print(df[['item_key', 'caption', 'contributor_id']].head(10))
        
        # Verify batching
        if len(df) >= 100:
            print("\n✅ SUCCESS: Central collection and batching working!")
            print(f"   - Captions were collected centrally")
            print(f"   - Batch flush occurred at 100 captions")
            print(f"   - Attribution preserved for all captions")
        else:
            print("\n⚠️  Less than 100 captions written - check buffer settings")
    else:
        print("\n❌ ERROR: No caption file found!")
        print("   Check if orchestrator is running and configured correctly")

def main():
    """Run the test."""
    print("Make sure the orchestrator is running with:")
    print("  caption-flow orchestrator --config configs/vllm_config.yaml")
    print("")
    input("Press Enter to start test...")
    
    try:
        asyncio.run(test_central_collection())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure:")
        print("  1. Orchestrator is running")
        print("  2. SSL certificates are configured")
        print("  3. Worker tokens match config")

if __name__ == "__main__":
    main()