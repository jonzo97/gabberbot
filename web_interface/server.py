#!/usr/bin/env python3
"""
Local Web Server - Standard Library Only
Self-contained web interface for GABBERBOT using only Python standard library
No FastAPI, uvicorn, or external dependencies required
"""

import json
import threading
import time
import socket
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional, List
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_shared.interfaces.synthesizer import MockSynthesizer
from cli_shared.ai.local_conversation_engine import create_local_conversation_engine
from cli_shared.analysis.local_audio_analyzer import create_local_audio_analyzer
from cli_shared.models.hardcore_models import HardcorePattern, SynthType

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-safe HTTP server"""
    daemon_threads = True

class GabberbotHandler(BaseHTTPRequestHandler):
    """HTTP request handler for GABBERBOT web interface"""
    
    def __init__(self, *args, **kwargs):
        self.server_instance = None
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Override to reduce noise in logs"""
        pass  # Silent logging
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/' or self.path == '/index.html':
                self.serve_main_interface()
            elif self.path == '/api/status':
                self.serve_status()
            elif self.path == '/api/patterns':
                self.serve_patterns()
            elif self.path.startswith('/static/') or self.path.endswith('.js'):
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
            elif self.path == '/api/analyze':
                self.handle_analyze(post_data)
            elif self.path == '/api/save':
                self.handle_save(post_data)
            elif self.path == '/api/load':
                self.handle_load(post_data)
            else:
                self.send_404()
        except Exception as e:
            self.send_error_response(f"POST error: {str(e)}")
    
    def serve_main_interface(self):
        """Serve the main web interface"""
        html_content = self.get_main_html()
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
                "uptime": int(time.time() - server.start_time)
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
    
    def handle_chat(self, post_data: str):
        """Handle chat messages"""
        try:
            data = json.loads(post_data)
            message = data.get('message', '')
            session_id = data.get('session_id', 'default')
            
            server = self.server.gabberbot_server
            
            # Process message through conversation engine
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    server.conversation_engine.process_message(message, session_id)
                )
                
                # Update current pattern if one was created
                if response.pattern:
                    server.current_pattern = response.pattern
                
                # Prepare response
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
                    } if response.pattern else None
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
                    
                    generate_response = {
                        "success": True,
                        "pattern": {
                            "name": response.pattern.name,
                            "bpm": response.pattern.bpm,
                            "genre": response.pattern.genre,
                            "synth_type": response.pattern.synth_type.value if response.pattern.synth_type else None,
                            "pattern_data": response.pattern.pattern_data
                        },
                        "message": response.response_text
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
    
    def handle_analyze(self, post_data: str):
        """Handle audio analysis requests"""
        try:
            server = self.server.gabberbot_server
            
            if not server.current_pattern:
                self.send_json_response({
                    "success": False,
                    "message": "No current pattern to analyze"
                })
                return
            
            # Generate audio for current pattern
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                audio = loop.run_until_complete(
                    server.synthesizer.play_pattern(server.current_pattern)
                )
                
                if audio is not None:
                    # Analyze the audio
                    analysis = loop.run_until_complete(
                        server.analyzer.analyze_audio(audio)
                    )
                    
                    kick_analysis = loop.run_until_complete(
                        server.analyzer.analyze_kick_dna(audio)
                    )
                    
                    analysis_response = {
                        "success": True,
                        "basic_analysis": {
                            "peak_level": analysis.peak_level,
                            "rms_level": analysis.rms_level,
                            "spectral_centroid": analysis.spectral_centroid,
                            "zero_crossing_rate": analysis.zero_crossing_rate,
                            "frequency_spectrum": analysis.frequency_spectrum
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
                else:
                    analysis_response = {
                        "success": False,
                        "message": "Failed to generate audio for analysis"
                    }
                
                self.send_json_response(analysis_response)
            finally:
                loop.close()
                
        except Exception as e:
            self.send_error_response(f"Analyze error: {str(e)}")
    
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
                
                load_response = {
                    "success": response.success,
                    "message": response.response_text,
                    "pattern": {
                        "name": response.pattern.name,
                        "bpm": response.pattern.bpm,
                        "genre": response.pattern.genre,
                        "synth_type": response.pattern.synth_type.value if response.pattern.synth_type else None,
                        "pattern_data": response.pattern.pattern_data
                    } if response.pattern else None
                }
                
                self.send_json_response(load_response)
            finally:
                loop.close()
                
        except Exception as e:
            self.send_error_response(f"Load error: {str(e)}")
    
    def serve_static_file(self):
        """Serve static files (placeholder)"""
        self.send_text_response("// Static file placeholder", "application/javascript")
    
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
    
    def send_text_response(self, content: str, content_type: str = "text/plain"):
        """Send text response"""
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Content-length', str(len(content.encode())))
        self.end_headers()
        self.wfile.write(content.encode())
    
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
    
    def get_main_html(self) -> str:
        """Generate main HTML interface"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔥 GABBERBOT - Hardcore Music Assistant</title>
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
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
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
            padding: 10px;
            background: #1a1a1a;
            border: 2px solid #ff0080;
            margin-bottom: 20px;
        }
        .header h1 {
            color: #ff0080;
            font-size: 2em;
            text-shadow: 0 0 10px #ff0080;
        }
        .status {
            color: #00ff80;
            margin: 10px 0;
        }
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
        .frequency-bars {
            display: flex;
            gap: 2px;
            margin: 10px 0;
            height: 60px;
            align-items: flex-end;
        }
        .freq-bar {
            flex: 1;
            background: linear-gradient(to top, #ff0080, #00ff80);
            min-height: 2px;
        }
        .log {
            font-size: 0.8em;
            color: #666;
            max-height: 150px;
            overflow-y: auto;
            background: #0a0a0a;
            padding: 10px;
            border: 1px solid #333;
            border-radius: 4px;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .loading {
            animation: pulse 1s infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 GABBERBOT 🔥</h1>
        <div class="status" id="status">Hardcore Music Assistant - System Operational</div>
    </div>
    
    <div class="container">
        <div class="panel">
            <h2>💬 Chat Interface</h2>
            <div class="chat-container">
                <div class="chat-history" id="chatHistory">
                    <div class="chat-message bot-message">
                        💥 GABBERBOT online! Ready to make some brutal hardcore music!<br>
                        Try: "Make a gabber kick at 180 BPM" or "Create industrial pattern"
                    </div>
                </div>
                <div class="chat-input">
                    <input type="text" id="chatInput" placeholder="Tell me what sound you want..." />
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h2>🎛️ Control Panel</h2>
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
            
            <div class="pattern-info" id="patternInfo">
                <h4>Current Pattern</h4>
                <div id="currentPatternDetails">No pattern loaded</div>
            </div>
            
            <div id="analysisResults" style="display: none;">
                <h3>📊 Audio Analysis</h3>
                <div class="frequency-bars" id="frequencyBars"></div>
                <div id="analysisDetails"></div>
            </div>
            
            <div class="log" id="systemLog">
                System initialized - Ready for hardcore music production!
            </div>
        </div>
    </div>

    <script>
        let currentPattern = null;
        
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
                } else {
                    appendMessage('bot', '❌ ' + data.message);
                    logMessage('Generation failed: ' + data.message);
                }
            } catch (error) {
                appendMessage('bot', '❌ Generation error: ' + error.message);
                logMessage('Generate error: ' + error.message);
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
        
        function updateAnalysisResults(analysis) {
            const container = document.getElementById('analysisResults');
            const bars = document.getElementById('frequencyBars');
            const details = document.getElementById('analysisDetails');
            
            // Update frequency bars
            bars.innerHTML = '';
            if (analysis.basic_analysis && analysis.basic_analysis.frequency_spectrum) {
                analysis.basic_analysis.frequency_spectrum.forEach(value => {
                    const bar = document.createElement('div');
                    bar.className = 'freq-bar';
                    bar.style.height = Math.max(2, value * 100) + '%';
                    bars.appendChild(bar);
                });
            }
            
            // Update analysis details
            if (analysis.kick_analysis) {
                const kick = analysis.kick_analysis;
                details.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9em;">
                        <div>Attack: ${kick.attack_time.toFixed(1)}ms</div>
                        <div>Sustain: ${(kick.sustain_level * 100).toFixed(1)}%</div>
                        <div>Decay: ${kick.decay_rate.toFixed(2)}</div>
                        <div>Fundamental: ${kick.fundamental_freq.toFixed(0)}Hz</div>
                        <div>Punch: ${(kick.punch_factor * 100).toFixed(0)}%</div>
                        <div>Rumble: ${(kick.rumble_factor * 100).toFixed(0)}%</div>
                    </div>
                `;
            }
            
            container.style.display = 'block';
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
                document.getElementById('status').textContent = 
                    `Status: ${data.status} | Uptime: ${data.uptime}s | Patterns: ${data.session_patterns}`;
            } catch (error) {
                document.getElementById('status').textContent = 'Status: Error connecting to server';
            }
        }, 5000);
        
        logMessage('Web interface initialized - Ready to make hardcore music!');
    </script>
</body>
</html>"""


class GabberbotServer:
    """Main GABBERBOT server class"""
    
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
        self.start_time = time.time()
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize GABBERBOT components"""
        print("🔥 Initializing GABBERBOT components...")
        
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
        
        print("🎵 GABBERBOT core systems operational!")
    
    def start_server(self):
        """Start the web server"""
        print(f"🚀 Starting GABBERBOT web server on http://{self.host}:{self.port}")
        
        try:
            # Create server
            self.server = ThreadingHTTPServer((self.host, self.port), GabberbotHandler)
            self.server.gabberbot_server = self  # Attach server instance
            
            self.running = True
            print(f"✅ Server running on http://{self.host}:{self.port}")
            print(f"💥 GABBERBOT web interface ready!")
            print(f"🎯 Try: 'Make a brutal gabber kick at 180 BPM'")
            print(f"⚡ Press Ctrl+C to stop")
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the web server"""
        if self.server:
            print("🛑 Stopping GABBERBOT server...")
            self.running = False
            self.server.shutdown()
            self.server.server_close()
            print("✅ Server stopped")
    
    def run_interactive(self):
        """Run server interactively"""
        if not self.start_server():
            return False
        
        try:
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n🛑 Shutting down GABBERBOT...")
            self.stop_server()
            
            # Stop synthesizer
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.synthesizer.stop())
            finally:
                loop.close()
            
            print("💥 GABBERBOT shutdown complete!")
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GABBERBOT Local Web Server')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('--test', action='store_true', help='Run quick test and exit')
    
    args = parser.parse_args()
    
    if args.test:
        print("🔥 GABBERBOT Local Web Server Test 🔥")
        print("=" * 50)
        
        server = GabberbotServer(args.host, args.port)
        print("✅ Server components initialized successfully")
        print("🎵 All systems operational - ready for full deployment!")
        return True
    
    # Run full server
    server = GabberbotServer(args.host, args.port)
    return server.run_interactive()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)