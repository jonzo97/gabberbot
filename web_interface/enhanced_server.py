#!/usr/bin/env python3
"""
Enhanced GABBERBOT Web Server
Full-featured web interface with audio playback, real-time controls, and pattern management
"""

import json
import threading
import time
import base64
import wave
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional, List
import asyncio
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_shared.interfaces.synthesizer import MockSynthesizer
from cli_shared.ai.local_conversation_engine import create_local_conversation_engine
from cli_shared.analysis.local_audio_analyzer import create_local_audio_analyzer
from cli_shared.models.hardcore_models import HardcorePattern, SynthType

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-safe HTTP server"""
    daemon_threads = True

class EnhancedGabberbotHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP request handler with audio support"""
    
    def log_message(self, format, *args):
        """Override to reduce noise in logs"""
        pass  # Silent logging
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/' or self.path == '/index.html':
                self.serve_enhanced_interface()
            elif self.path == '/api/status':
                self.serve_status()
            elif self.path == '/api/patterns':
                self.serve_patterns()
            elif self.path.startswith('/api/audio/'):
                self.serve_audio()
            elif self.path.startswith('/static/'):
                self.serve_static_file()
            else:
                self.send_404()
        except Exception as e:
            self.send_error_response(f"GET error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            if self.path == '/api/chat':
                self.handle_chat(post_data)
            elif self.path == '/api/generate':
                self.handle_generate(post_data)
            elif self.path == '/api/play':
                self.handle_play(post_data)
            elif self.path == '/api/stop':
                self.handle_stop(post_data)
            elif self.path == '/api/analyze':
                self.handle_analyze(post_data)
            elif self.path == '/api/export':
                self.handle_export(post_data)
            elif self.path == '/api/save':
                self.handle_save(post_data)
            elif self.path == '/api/load':
                self.handle_load(post_data)
            else:
                self.send_404()
        except Exception as e:
            self.send_error_response(f"POST error: {str(e)}")
    
    def serve_enhanced_interface(self):
        """Serve the enhanced web interface with audio controls"""
        html_content = self.get_enhanced_html()
        self.send_html_response(html_content)
    
    def serve_status(self):
        """Serve system status"""
        try:
            server = self.server.gabberbot_server
            status = {
                "status": "operational",
                "synthesizer_state": server.synthesizer.get_state().value,
                "current_pattern": server.current_pattern.name if server.current_pattern else None,
                "session_patterns": len(server.conversation_engine.get_session_patterns()),
                "conversation_history": len(server.conversation_engine.get_conversation_history()),
                "uptime": int(time.time() - server.start_time),
                "is_playing": server.is_playing,
                "current_bpm": getattr(server.current_pattern, 'bpm', 180) if server.current_pattern else 180
            }
            self.send_json_response(status)
        except Exception as e:
            self.send_error_response(f"Status error: {str(e)}")
    
    def serve_patterns(self):
        """Serve saved patterns"""
        try:
            server = self.server.gabberbot_server
            patterns = {}
            for name, pattern in server.conversation_engine.get_session_patterns().items():
                patterns[name] = {
                    "name": pattern.name,
                    "bpm": pattern.bpm,
                    "genre": pattern.genre,
                    "synth_type": pattern.synth_type.value if pattern.synth_type else None,
                    "pattern_data": pattern.pattern_data
                }
            self.send_json_response(patterns)
        except Exception as e:
            self.send_error_response(f"Patterns error: {str(e)}")
    
    def serve_audio(self):
        """Serve audio file for current pattern"""
        try:
            server = self.server.gabberbot_server
            if not server.current_pattern or not hasattr(server, 'current_audio'):
                self.send_404()
                return
            
            # Generate WAV file from audio data
            wav_data = self.create_wav_data(server.current_audio, server.synthesizer.sample_rate)
            
            self.send_response(200)
            self.send_header('Content-type', 'audio/wav')
            self.send_header('Content-length', str(len(wav_data)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(wav_data)
            
        except Exception as e:
            self.send_error_response(f"Audio error: {str(e)}")
    
    def handle_chat(self, post_data: str):
        """Handle chat messages"""
        try:
            data = json.loads(post_data)
            message = data.get('message', '')
            session_id = data.get('session_id', 'default')
            
            server = self.server.gabberbot_server
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    server.conversation_engine.process_message(message, session_id)
                )
                
                # Update current pattern if one was created
                if response.pattern:
                    server.current_pattern = response.pattern
                    # Generate audio for the new pattern
                    audio = loop.run_until_complete(
                        server.synthesizer.play_pattern(response.pattern)
                    )
                    if audio is not None:
                        server.current_audio = audio
                
                chat_response = {
                    "response": response.response_text,
                    "intent": response.intent.value,
                    "confidence": response.confidence,
                    "success": response.success,
                    "pattern": {
                        "name": response.pattern.name,
                        "bpm": response.pattern.bpm,
                        "genre": response.pattern.genre,
                        "synth_type": response.pattern.synth_type.value if response.pattern.synth_type else None,
                        "pattern_data": response.pattern.pattern_data
                    } if response.pattern else None,
                    "has_audio": hasattr(server, 'current_audio')
                }
                
                self.send_json_response(chat_response)
            finally:
                loop.close()
                
        except Exception as e:
            self.send_error_response(f"Chat error: {str(e)}")
    
    def handle_generate(self, post_data: str):
        """Handle pattern generation requests"""
        try:
            data = json.loads(post_data)
            genre = data.get('genre', 'gabber')
            bpm = data.get('bpm', 180)
            style = data.get('style', 'brutal')
            
            server = self.server.gabberbot_server
            
            # Generate pattern through conversation engine
            prompt = f"Make a {style} {genre} pattern at {bpm} BPM"
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    server.conversation_engine.process_message(prompt)
                )
                
                if response.pattern:
                    server.current_pattern = response.pattern
                    # Generate audio
                    audio = loop.run_until_complete(
                        server.synthesizer.play_pattern(response.pattern)
                    )
                    if audio is not None:
                        server.current_audio = audio
                    
                    generate_response = {
                        "success": True,
                        "pattern": {
                            "name": response.pattern.name,
                            "bpm": response.pattern.bpm,
                            "genre": response.pattern.genre,
                            "synth_type": response.pattern.synth_type.value if response.pattern.synth_type else None,
                            "pattern_data": response.pattern.pattern_data
                        },
                        "message": response.response_text,
                        "has_audio": hasattr(server, 'current_audio')
                    }
                else:
                    generate_response = {
                        "success": False,
                        "message": "Pattern generation failed"
                    }
                
                self.send_json_response(generate_response)
            finally:
                loop.close()
                
        except Exception as e:
            self.send_error_response(f"Generate error: {str(e)}")
    
    def handle_play(self, post_data: str):
        """Handle play requests"""
        try:
            server = self.server.gabberbot_server
            
            if not server.current_pattern:
                self.send_json_response({
                    "success": False,
                    "message": "No pattern to play"
                })
                return
            
            # Set playing state
            server.is_playing = True
            
            # Generate fresh audio if needed
            if not hasattr(server, 'current_audio'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    audio = loop.run_until_complete(
                        server.synthesizer.play_pattern(server.current_pattern)
                    )
                    if audio is not None:
                        server.current_audio = audio
                finally:
                    loop.close()
            
            self.send_json_response({
                "success": True,
                "message": f"Playing {server.current_pattern.name}",
                "pattern_name": server.current_pattern.name,
                "bpm": server.current_pattern.bpm,
                "duration": len(server.current_audio) / server.synthesizer.sample_rate if hasattr(server, 'current_audio') else 1.0
            })
            
        except Exception as e:
            self.send_error_response(f"Play error: {str(e)}")
    
    def handle_stop(self, post_data: str):
        """Handle stop requests"""
        try:
            server = self.server.gabberbot_server
            server.is_playing = False
            
            self.send_json_response({
                "success": True,
                "message": "Stopped playback"
            })
            
        except Exception as e:
            self.send_error_response(f"Stop error: {str(e)}")
    
    def handle_analyze(self, post_data: str):
        """Handle audio analysis requests"""
        try:
            server = self.server.gabberbot_server
            
            if not hasattr(server, 'current_audio'):
                self.send_json_response({
                    "success": False,
                    "message": "No audio to analyze"
                })
                return
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Analyze the audio
                analysis = loop.run_until_complete(
                    server.analyzer.analyze_audio(server.current_audio)
                )
                
                kick_analysis = loop.run_until_complete(
                    server.analyzer.analyze_kick_dna(server.current_audio)
                )
                
                analysis_response = {
                    "success": True,
                    "basic_analysis": {
                        "peak_level": analysis.peak,
                        "rms_level": analysis.rms,
                        "spectral_centroid": analysis.spectral_centroid
                    },
                    "kick_analysis": {
                        "attack_time": kick_analysis.attack_time,
                        "sustain_level": kick_analysis.sustain_level,
                        "decay_rate": kick_analysis.decay_rate,
                        "fundamental_freq": kick_analysis.fundamental_freq,
                        "harmonic_ratio": kick_analysis.harmonic_ratio,
                        "punch_factor": kick_analysis.punch_factor,
                        "rumble_factor": kick_analysis.rumble_factor
                    }
                }
                
                self.send_json_response(analysis_response)
            finally:
                loop.close()
                
        except Exception as e:
            self.send_error_response(f"Analyze error: {str(e)}")
    
    def handle_export(self, post_data: str):
        """Handle audio export requests"""
        try:
            data = json.loads(post_data)
            format_type = data.get('format', 'wav')
            
            server = self.server.gabberbot_server
            
            if not hasattr(server, 'current_audio'):
                self.send_json_response({
                    "success": False,
                    "message": "No audio to export"
                })
                return
            
            # Create WAV data
            wav_data = self.create_wav_data(server.current_audio, server.synthesizer.sample_rate)
            
            # Encode as base64 for download
            wav_b64 = base64.b64encode(wav_data).decode('utf-8')
            
            pattern_name = server.current_pattern.name if server.current_pattern else "gabber_pattern"
            
            self.send_json_response({
                "success": True,
                "filename": f"{pattern_name}.wav",
                "data": wav_b64,
                "size": len(wav_data)
            })
            
        except Exception as e:
            self.send_error_response(f"Export error: {str(e)}")
    
    def handle_save(self, post_data: str):
        """Handle pattern save requests"""
        try:
            data = json.loads(post_data)
            pattern_name = data.get('name', f'pattern_{int(time.time())}')
            
            server = self.server.gabberbot_server
            
            if not server.current_pattern:
                self.send_json_response({
                    "success": False,
                    "message": "No current pattern to save"
                })
                return
            
            # Save through conversation engine
            prompt = f"save as {pattern_name}"
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    server.conversation_engine.process_message(prompt)
                )
                
                save_response = {
                    "success": response.success,
                    "message": response.response_text
                }
                
                self.send_json_response(save_response)
            finally:
                loop.close()
                
        except Exception as e:
            self.send_error_response(f"Save error: {str(e)}")
    
    def handle_load(self, post_data: str):
        """Handle pattern load requests"""
        try:
            data = json.loads(post_data)
            pattern_name = data.get('name', '')
            
            if not pattern_name:
                self.send_json_response({
                    "success": False,
                    "message": "Pattern name required"
                })
                return
            
            server = self.server.gabberbot_server
            
            # Load through conversation engine
            prompt = f"load {pattern_name}"
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    server.conversation_engine.process_message(prompt)
                )
                
                if response.pattern:
                    server.current_pattern = response.pattern
                    # Generate audio for loaded pattern
                    audio = loop.run_until_complete(
                        server.synthesizer.play_pattern(response.pattern)
                    )
                    if audio is not None:
                        server.current_audio = audio
                
                load_response = {
                    "success": response.success,
                    "message": response.response_text,
                    "pattern": {
                        "name": response.pattern.name,
                        "bpm": response.pattern.bpm,
                        "genre": response.pattern.genre,
                        "synth_type": response.pattern.synth_type.value if response.pattern.synth_type else None,
                        "pattern_data": response.pattern.pattern_data
                    } if response.pattern else None,
                    "has_audio": hasattr(server, 'current_audio')
                }
                
                self.send_json_response(load_response)
            finally:
                loop.close()
                
        except Exception as e:
            self.send_error_response(f"Load error: {str(e)}")
    
    def create_wav_data(self, audio_samples: np.ndarray, sample_rate: int) -> bytes:
        """Create WAV file data from audio samples"""
        # Ensure audio is in correct format
        audio_samples = np.array(audio_samples, dtype=np.float32)
        
        # Convert to 16-bit integers
        audio_int16 = (audio_samples * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()
    
    def send_html_response(self, content: str):
        """Send HTML response"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-length', str(len(content.encode())))
        self.end_headers()
        self.wfile.write(content.encode())
    
    def send_json_response(self, data: Dict[str, Any]):
        """Send JSON response"""
        json_content = json.dumps(data, indent=2)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-length', str(len(json_content.encode())))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json_content.encode())
    
    def send_404(self):
        """Send 404 response"""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>404 - Not Found</h1><p>GABBERBOT endpoint not found</p>')
    
    def send_error_response(self, error: str):
        """Send error response"""
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        error_data = json.dumps({"error": error, "success": False})
        self.send_header('Content-length', str(len(error_data.encode())))
        self.end_headers()
        self.wfile.write(error_data.encode())
    
    def get_enhanced_html(self) -> str:
        """Generate enhanced HTML interface with audio controls"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔥 GABBERBOT - Enhanced Hardcore Music Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            padding: 20px;
            line-height: 1.4;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr 300px;
            gap: 20px;
            height: 90vh;
        }
        .panel {
            background: #111;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 20px;
            overflow-y: auto;
        }
        .header {
            grid-column: 1 / -1;
            text-align: center;
            padding: 15px;
            background: #1a1a1a;
            border: 2px solid #ff0080;
            margin-bottom: 20px;
            position: relative;
        }
        .header h1 {
            color: #ff0080;
            font-size: 2em;
            text-shadow: 0 0 10px #ff0080;
        }
        .status-bar {
            position: absolute;
            top: 10px;
            right: 20px;
            color: #00ff80;
            font-size: 0.9em;
        }
        
        /* Audio Controls */
        .audio-controls {
            background: #222;
            border: 2px solid #ff0080;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }
        .audio-controls h3 {
            color: #ff0080;
            margin-bottom: 15px;
            text-align: center;
        }
        .play-controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        .play-btn, .stop-btn {
            background: #ff0080;
            color: #000;
            border: none;
            padding: 12px 24px;
            border-radius: 50px;
            font-weight: bold;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
        }
        .play-btn:hover, .stop-btn:hover {
            background: #00ff80;
            box-shadow: 0 0 15px #00ff80;
        }
        .play-btn:disabled, .stop-btn:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .audio-info {
            text-align: center;
            color: #00ff80;
            margin-top: 10px;
        }
        .waveform {
            height: 60px;
            background: #0d0d0d;
            border: 1px solid #555;
            border-radius: 4px;
            margin: 10px 0;
            position: relative;
            overflow: hidden;
        }
        .waveform-bar {
            position: absolute;
            bottom: 0;
            background: linear-gradient(to top, #ff0080, #00ff80);
            width: 2px;
            margin-right: 1px;
            transition: height 0.1s;
        }
        
        /* Chat Interface */
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .chat-history {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 15px;
            padding: 10px;
            background: #0d0d0d;
            border: 1px solid #333;
            border-radius: 4px;
        }
        .chat-message {
            margin: 10px 0;
            padding: 8px;
            border-left: 3px solid #666;
            padding-left: 12px;
        }
        .user-message {
            border-left-color: #00ff00;
            color: #00ff00;
        }
        .bot-message {
            border-left-color: #ff0080;
            color: #ff0080;
        }
        .chat-input {
            display: flex;
            gap: 10px;
        }
        
        /* Controls */
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        .control-group {
            border: 1px solid #444;
            padding: 15px;
            border-radius: 4px;
        }
        .control-group h3 {
            color: #ff0080;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        input, button, select {
            background: #222;
            border: 1px solid #555;
            color: #00ff00;
            padding: 8px 12px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }
        input {
            flex: 1;
        }
        button {
            background: #333;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:hover {
            background: #ff0080;
            color: #000;
            box-shadow: 0 0 10px #ff0080;
        }
        
        /* Pattern Info */
        .pattern-info {
            background: #0d0d0d;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #555;
            border-radius: 4px;
        }
        .pattern-info h4 {
            color: #00ff80;
            margin-bottom: 5px;
        }
        
        /* Analysis Results */
        .analysis-results {
            background: #0d0d0d;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #555;
            border-radius: 4px;
        }
        .analysis-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 0.9em;
        }
        .analysis-item {
            padding: 5px;
            background: #1a1a1a;
            border-radius: 3px;
        }
        
        /* Export Controls */
        .export-controls {
            margin-top: 20px;
            padding: 15px;
            background: #1a1a1a;
            border-radius: 4px;
        }
        .export-btn {
            background: #00ff80;
            color: #000;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
        }
        
        /* Sidebar */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .sidebar .panel {
            padding: 15px;
        }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .container {
                grid-template-columns: 1fr 1fr;
            }
            .sidebar {
                grid-column: 1 / -1;
                flex-direction: row;
                gap: 10px;
            }
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
            .controls {
                grid-template-columns: 1fr;
            }
        }
        
        /* Animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .playing {
            animation: pulse 1s infinite;
        }
        .loading {
            animation: pulse 0.5s infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 GABBERBOT 🔥</h1>
        <div class="status-bar" id="statusBar">System Operational</div>
    </div>
    
    <div class="container">
        <!-- Chat Panel -->
        <div class="panel">
            <h2>💬 Chat Interface</h2>
            <div class="chat-container">
                <div class="chat-history" id="chatHistory">
                    <div class="chat-message bot-message">
                        💥 GABBERBOT Enhanced Interface Online! Ready to make brutal hardcore music!<br>
                        🎵 Try: "Make a gabber kick at 180 BPM" or "Create industrial pattern"<br>
                        🔥 Use the audio controls to play and analyze your patterns!
                    </div>
                </div>
                <div class="chat-input">
                    <input type="text" id="chatInput" placeholder="Tell me what hardcore sound you want..." />
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
        
        <!-- Control Panel -->
        <div class="panel">
            <h2>🎛️ Control Panel</h2>
            
            <!-- Audio Controls -->
            <div class="audio-controls">
                <h3>🔊 Audio Playback</h3>
                <div class="play-controls">
                    <button class="play-btn" id="playBtn" onclick="playAudio()" disabled>▶ PLAY</button>
                    <button class="stop-btn" id="stopBtn" onclick="stopAudio()" disabled>⏹ STOP</button>
                </div>
                <div class="waveform" id="waveform"></div>
                <div class="audio-info" id="audioInfo">No pattern loaded</div>
            </div>
            
            <!-- Generation Controls -->
            <div class="controls">
                <div class="control-group">
                    <h3>Generate</h3>
                    <select id="genreSelect">
                        <option value="gabber">Gabber</option>
                        <option value="industrial">Industrial</option>
                        <option value="hardcore">Hardcore</option>
                    </select>
                    <input type="number" id="bpmInput" value="180" min="130" max="250" />
                    <button onclick="generatePattern()">Generate Pattern</button>
                </div>
                <div class="control-group">
                    <h3>Pattern</h3>
                    <button onclick="analyzePattern()">Analyze Audio</button>
                    <input type="text" id="saveNameInput" placeholder="Pattern name" />
                    <button onclick="savePattern()">Save Pattern</button>
                </div>
            </div>
            
            <!-- Pattern Info -->
            <div class="pattern-info" id="patternInfo">
                <h4>Current Pattern</h4>
                <div id="currentPatternDetails">No pattern loaded</div>
            </div>
            
            <!-- Export Controls -->
            <div class="export-controls">
                <button class="export-btn" onclick="exportAudio()">💾 Export WAV File</button>
            </div>
        </div>
        
        <!-- Sidebar -->
        <div class="sidebar">
            <!-- Analysis Results -->
            <div class="panel">
                <h3>📊 Audio Analysis</h3>
                <div class="analysis-results" id="analysisResults" style="display: none;">
                    <div class="analysis-grid" id="analysisGrid"></div>
                </div>
                <div id="noAnalysis" style="text-align: center; color: #666;">
                    Generate a pattern and analyze it to see detailed audio metrics
                </div>
            </div>
            
            <!-- System Log -->
            <div class="panel">
                <h3>📋 System Log</h3>
                <div id="systemLog" style="max-height: 200px; overflow-y: auto; font-size: 0.8em; color: #666;">
                    System initialized - Enhanced interface ready for hardcore music production!
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentPattern = null;
        let isPlaying = false;
        let audioContext = null;
        let audioBuffer = null;
        let audioSource = null;
        
        // Initialize audio context
        function initAudio() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            appendMessage('user', message);
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message, session_id: 'web'})
                });
                const data = await response.json();
                
                appendMessage('bot', data.response);
                
                if (data.pattern) {
                    currentPattern = data.pattern;
                    updatePatternInfo(data.pattern);
                    
                    if (data.has_audio) {
                        document.getElementById('playBtn').disabled = false;
                        updateAudioInfo(data.pattern);
                    }
                }
                
                logMessage(`Intent: ${data.intent}, Confidence: ${(data.confidence * 100).toFixed(1)}%`);
            } catch (error) {
                appendMessage('bot', '❌ Error: ' + error.message);
                logMessage('Chat error: ' + error.message);
            }
        }
        
        async function generatePattern() {
            const genre = document.getElementById('genreSelect').value;
            const bpm = parseInt(document.getElementById('bpmInput').value);
            
            logMessage(`Generating ${genre} pattern at ${bpm} BPM...`);
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({genre: genre, bpm: bpm, style: 'brutal'})
                });
                const data = await response.json();
                
                if (data.success) {
                    currentPattern = data.pattern;
                    updatePatternInfo(data.pattern);
                    appendMessage('bot', data.message);
                    logMessage('Pattern generated successfully');
                    
                    if (data.has_audio) {
                        document.getElementById('playBtn').disabled = false;
                        updateAudioInfo(data.pattern);
                    }
                } else {
                    appendMessage('bot', '❌ ' + data.message);
                    logMessage('Generation failed: ' + data.message);
                }
            } catch (error) {
                appendMessage('bot', '❌ Generation error: ' + error.message);
                logMessage('Generate error: ' + error.message);
            }
        }
        
        async function playAudio() {
            if (!currentPattern) {
                logMessage('No pattern to play');
                return;
            }
            
            try {
                initAudio();
                
                // Request play from server
                const response = await fetch('/api/play', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
                const data = await response.json();
                
                if (data.success) {
                    // Fetch and play audio
                    const audioResponse = await fetch('/api/audio/current');
                    const arrayBuffer = await audioResponse.arrayBuffer();
                    audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    
                    // Play audio
                    audioSource = audioContext.createBufferSource();
                    audioSource.buffer = audioBuffer;
                    audioSource.connect(audioContext.destination);
                    audioSource.start();
                    
                    // Update UI
                    isPlaying = true;
                    document.getElementById('playBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('playBtn').classList.add('playing');
                    
                    // Auto-stop when finished
                    audioSource.onended = () => {
                        stopAudio();
                    };
                    
                    logMessage(`Playing: ${data.pattern_name} (${data.duration.toFixed(1)}s)`);
                } else {
                    logMessage('Play failed: ' + data.message);
                }
            } catch (error) {
                logMessage('Play error: ' + error.message);
                stopAudio();
            }
        }
        
        async function stopAudio() {
            try {
                if (audioSource) {
                    audioSource.stop();
                    audioSource = null;
                }
                
                await fetch('/api/stop', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
                
                // Update UI
                isPlaying = false;
                document.getElementById('playBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('playBtn').classList.remove('playing');
                
                logMessage('Stopped playback');
            } catch (error) {
                logMessage('Stop error: ' + error.message);
            }
        }
        
        async function analyzePattern() {
            if (!currentPattern) {
                appendMessage('bot', '❌ No pattern to analyze! Generate one first.');
                return;
            }
            
            logMessage('Analyzing pattern audio...');
            
            try {
                const response = await fetch('/api/analyze', {method: 'POST'});
                const data = await response.json();
                
                if (data.success) {
                    updateAnalysisResults(data);
                    appendMessage('bot', '📊 Audio analysis complete!');
                    logMessage('Analysis completed successfully');
                } else {
                    appendMessage('bot', '❌ Analysis failed: ' + data.message);
                    logMessage('Analysis failed: ' + data.message);
                }
            } catch (error) {
                appendMessage('bot', '❌ Analysis error: ' + error.message);
                logMessage('Analyze error: ' + error.message);
            }
        }
        
        async function exportAudio() {
            if (!currentPattern) {
                logMessage('No pattern to export');
                return;
            }
            
            try {
                const response = await fetch('/api/export', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({format: 'wav'})
                });
                const data = await response.json();
                
                if (data.success) {
                    // Create download link
                    const blob = new Blob([Uint8Array.from(atob(data.data), c => c.charCodeAt(0))], 
                                         {type: 'audio/wav'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = data.filename;
                    a.click();
                    URL.revokeObjectURL(url);
                    
                    logMessage(`Exported: ${data.filename} (${(data.size/1024).toFixed(1)} KB)`);
                } else {
                    logMessage('Export failed: ' + data.message);
                }
            } catch (error) {
                logMessage('Export error: ' + error.message);
            }
        }
        
        async function savePattern() {
            if (!currentPattern) {
                appendMessage('bot', '❌ No pattern to save!');
                return;
            }
            
            const name = document.getElementById('saveNameInput').value.trim() || `pattern_${Date.now()}`;
            
            try {
                const response = await fetch('/api/save', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: name})
                });
                const data = await response.json();
                
                appendMessage('bot', data.message);
                if (data.success) {
                    document.getElementById('saveNameInput').value = '';
                    logMessage(`Pattern saved as: ${name}`);
                }
            } catch (error) {
                appendMessage('bot', '❌ Save error: ' + error.message);
                logMessage('Save error: ' + error.message);
            }
        }
        
        function appendMessage(sender, message) {
            const history = document.getElementById('chatHistory');
            const div = document.createElement('div');
            div.className = `chat-message ${sender}-message`;
            div.innerHTML = message.replace(/\\n/g, '<br>');
            history.appendChild(div);
            history.scrollTop = history.scrollHeight;
        }
        
        function updatePatternInfo(pattern) {
            const details = document.getElementById('currentPatternDetails');
            details.innerHTML = `
                <strong>${pattern.name}</strong><br>
                Genre: ${pattern.genre}<br>
                BPM: ${pattern.bpm}<br>
                Synth: ${pattern.synth_type || 'default'}<br>
                <div style="font-family: monospace; font-size: 0.8em; color: #666; margin-top: 5px;">
                    ${pattern.pattern_data || 'No pattern data'}
                </div>
            `;
        }
        
        function updateAudioInfo(pattern) {
            const info = document.getElementById('audioInfo');
            info.innerHTML = `Ready to play: ${pattern.name} @ ${pattern.bpm} BPM`;
        }
        
        function updateAnalysisResults(analysis) {
            const container = document.getElementById('analysisResults');
            const grid = document.getElementById('analysisGrid');
            const noAnalysis = document.getElementById('noAnalysis');
            
            // Update analysis display
            if (analysis.kick_analysis) {
                const kick = analysis.kick_analysis;
                const basic = analysis.basic_analysis;
                
                grid.innerHTML = `
                    <div class="analysis-item">
                        <strong>Peak Level:</strong><br>
                        ${basic.peak_level.toFixed(3)}
                    </div>
                    <div class="analysis-item">
                        <strong>RMS Level:</strong><br>
                        ${basic.rms_level.toFixed(3)}
                    </div>
                    <div class="analysis-item">
                        <strong>Attack Time:</strong><br>
                        ${kick.attack_time.toFixed(1)}ms
                    </div>
                    <div class="analysis-item">
                        <strong>Punch Factor:</strong><br>
                        ${(kick.punch_factor * 100).toFixed(0)}%
                    </div>
                    <div class="analysis-item">
                        <strong>Fundamental:</strong><br>
                        ${kick.fundamental_freq.toFixed(0)}Hz
                    </div>
                    <div class="analysis-item">
                        <strong>Rumble Factor:</strong><br>
                        ${(kick.rumble_factor * 100).toFixed(0)}%
                    </div>
                `;
            }
            
            container.style.display = 'block';
            noAnalysis.style.display = 'none';
        }
        
        function logMessage(message) {
            const log = document.getElementById('systemLog');
            const time = new Date().toLocaleTimeString();
            log.innerHTML += `\\n[${time}] ${message}`;
            log.scrollTop = log.scrollHeight;
        }
        
        // Handle Enter key in chat input
        document.getElementById('chatInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Update status periodically
        setInterval(async function() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('statusBar').textContent = 
                    `Status: ${data.status} | Uptime: ${data.uptime}s | Patterns: ${data.session_patterns} | Playing: ${data.is_playing ? 'YES' : 'NO'}`;
            } catch (error) {
                document.getElementById('statusBar').textContent = 'Status: Error connecting to server';
            }
        }, 2000);
        
        logMessage('Enhanced GABBERBOT interface initialized - Ready to destroy sound systems! 🔥');
    </script>
</body>
</html>"""


class EnhancedGabberbotServer:
    """Enhanced GABBERBOT server with full audio support"""
    
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False
        
        # Initialize components
        self.synthesizer = None
        self.conversation_engine = None
        self.analyzer = None
        self.current_pattern = None
        self.current_audio = None
        self.is_playing = False
        self.start_time = time.time()
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize GABBERBOT components"""
        print("🔥 Initializing Enhanced GABBERBOT components...")
        
        # Initialize synthesizer
        self.synthesizer = MockSynthesizer()
        print("✅ Mock synthesizer initialized")
        
        # Initialize conversation engine
        self.conversation_engine = create_local_conversation_engine(self.synthesizer)
        print("✅ Local conversation engine initialized")
        
        # Initialize audio analyzer
        self.analyzer = create_local_audio_analyzer()
        print("✅ Local audio analyzer initialized")
        
        # Start synthesizer
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.synthesizer.start())
            print("✅ Synthesizer started")
        finally:
            loop.close()
        
        print("🎵 Enhanced GABBERBOT core systems operational!")
    
    def start_server(self):
        """Start the enhanced web server"""
        print(f"🚀 Starting Enhanced GABBERBOT web server on http://{self.host}:{self.port}")
        
        try:
            # Create server
            self.server = ThreadingHTTPServer((self.host, self.port), EnhancedGabberbotHandler)
            self.server.gabberbot_server = self  # Attach server instance
            
            self.running = True
            print(f"✅ Enhanced server running on http://{self.host}:{self.port}")
            print(f"💥 GABBERBOT enhanced web interface ready!")
            print(f"🎵 Features: Audio playback, real-time analysis, pattern export")
            print(f"⚡ Press Ctrl+C to stop")
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start enhanced server: {e}")
            return False
    
    def stop_server(self):
        """Stop the enhanced web server"""
        if self.server:
            print("🛑 Stopping Enhanced GABBERBOT server...")
            self.running = False
            self.server.shutdown()
            self.server.server_close()
            print("✅ Enhanced server stopped")
    
    def run_interactive(self):
        """Run server interactively"""
        if not self.start_server():
            return False
        
        try:
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n🛑 Shutting down Enhanced GABBERBOT...")
            self.stop_server()
            
            # Stop synthesizer
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.synthesizer.stop())
            finally:
                loop.close()
            
            print("💥 Enhanced GABBERBOT shutdown complete!")
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced GABBERBOT Web Server')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('--test', action='store_true', help='Run quick test and exit')
    
    args = parser.parse_args()
    
    if args.test:
        print("🔥 Enhanced GABBERBOT Web Server Test 🔥")
        print("=" * 50)
        
        server = EnhancedGabberbotServer(args.host, args.port)
        print("✅ Enhanced server components initialized successfully")
        print("🎵 All systems operational - ready for enhanced deployment!")
        return True
    
    # Run full enhanced server
    server = EnhancedGabberbotServer(args.host, args.port)
    return server.run_interactive()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)