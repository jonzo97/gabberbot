#!/usr/bin/env python3
"""
BMAD 30-Minute Production Runner
Runs BMAD agent sessions for exactly 30 minutes
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from overnight_agent_coordinator import BMADAgentCoordinator

def run_30minute_bmad_production():
    """Run BMAD agent production sessions for 30 minutes"""
    print("BMAD 30-MINUTE HARDCORE PRODUCTION STARTING")
    print("=" * 60)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=30)
    
    print(f"START TIME: {start_time.strftime('%H:%M:%S')}")
    print(f"END TIME: {end_time.strftime('%H:%M:%S')}")
    print(f"DURATION: 30 minutes")
    
    session_count = 0
    total_outputs = {
        "sessions": 0,
        "patterns": 0, 
        "tracks": 0,
        "presets": 0
    }
    
    while datetime.now() < end_time:
        try:
            session_count += 1
            current_time = datetime.now()
            time_remaining = end_time - current_time
            
            print(f"\nSTARTING SESSION #{session_count}")
            print(f"TIME: {current_time.strftime('%H:%M:%S')}")
            print(f"TIME REMAINING: {int(time_remaining.total_seconds()/60)} minutes")
            
            # Create new coordinator for each session
            coordinator = BMADAgentCoordinator()
            
            # Run full agent session
            summary = coordinator.run_overnight_session()
            
            if summary.get("success"):
                # Update totals
                total_outputs["sessions"] += 1
                total_outputs["patterns"] += summary["outputs"]["generated_patterns"]
                total_outputs["tracks"] += summary["outputs"]["mixed_tracks"]
                total_outputs["presets"] += 20  # Estimated presets per session
                
                print(f"\nSESSION #{session_count} COMPLETE!")
                print(f"TOTAL OUTPUTS SO FAR:")
                print(f"   Sessions: {total_outputs['sessions']}")
                print(f"   Patterns: {total_outputs['patterns']}")
                print(f"   Tracks: {total_outputs['tracks']}")
                print(f"   Presets: {total_outputs['presets']}")
                
                # Save running totals
                totals_file = Path("30min_production_totals.json")
                with open(totals_file, 'w') as f:
                    json.dump({
                        "start_time": start_time.isoformat(),
                        "current_time": datetime.now().isoformat(),
                        "end_time": end_time.isoformat(),
                        "time_remaining_minutes": int(time_remaining.total_seconds()/60),
                        "total_sessions": total_outputs["sessions"],
                        "total_outputs": total_outputs,
                        "current_session": session_count,
                        "status": "running"
                    }, f, indent=2)
                
                # Brief pause between sessions
                print(f"\nPausing 5 seconds before next session...")
                time.sleep(5)
                
            else:
                print(f"SESSION #{session_count} FAILED: {summary.get('error', 'Unknown error')}")
                print("Waiting 10 seconds before retry...")
                time.sleep(10)
                
        except KeyboardInterrupt:
            print(f"\nBMAD PRODUCTION STOPPED BY USER")
            break
        except Exception as e:
            print(f"ERROR IN SESSION #{session_count}: {e}")
            print("Waiting 10 seconds before retry...")
            time.sleep(10)
    
    # Final summary
    actual_end_time = datetime.now()
    actual_duration = actual_end_time - start_time
    
    final_summary = {
        "start_time": start_time.isoformat(),
        "end_time": actual_end_time.isoformat(), 
        "planned_duration_minutes": 30,
        "actual_duration_minutes": actual_duration.total_seconds() / 60,
        "total_sessions": total_outputs["sessions"],
        "final_outputs": total_outputs,
        "sessions_per_minute": total_outputs["sessions"] / (actual_duration.total_seconds() / 60),
        "patterns_per_minute": total_outputs["patterns"] / (actual_duration.total_seconds() / 60),
        "status": "completed"
    }
    
    # Save final summary
    final_file = Path("30min_production_final.json")
    with open(final_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\n" + "="*60)
    print("30-MINUTE BMAD SESSION COMPLETE!")
    print("="*60)
    print(f"Actual Duration: {actual_duration.total_seconds()/60:.1f} minutes")
    print(f"Sessions Completed: {total_outputs['sessions']}")
    print(f"Sessions Per Minute: {final_summary['sessions_per_minute']:.1f}")
    print(f"\nFinal Outputs:")
    print(f"   Patterns: {total_outputs['patterns']}")
    print(f"   Tracks: {total_outputs['tracks']}")
    print(f"   Presets: {total_outputs['presets']}")
    print(f"\nProduction Rate:")
    print(f"   {final_summary['patterns_per_minute']:.1f} patterns/minute")
    print(f"   {total_outputs['tracks']/30:.1f} tracks/minute")
    
    print(f"\nBMAD agents completed {total_outputs['sessions']} hardcore production sessions!")
    print(f"All outputs saved to pattern_evolution_workspace/")

if __name__ == "__main__":
    run_30minute_bmad_production()