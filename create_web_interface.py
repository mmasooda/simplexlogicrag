#!/usr/bin/env python3
"""
Create a simple web interface for the Simplex RAG system.
"""

import sys
import os
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv('/root/.env')

# Add simplexrag to path
sys.path.insert(0, '/root/simplexrag')

from flask import Flask, request, jsonify, render_template_string
from simplex_rag.orchestrator import SimplexRAGOrchestrator

app = Flask(__name__)

# Initialize the RAG system
try:
    orch = SimplexRAGOrchestrator()
    print("✅ Simplex RAG system initialized")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    orch = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simplex RAG System</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .container { display: flex; gap: 20px; }
        .panel { flex: 1; border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
        .panel h2 { margin-top: 0; color: #333; }
        textarea { width: 100%; height: 120px; margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #005a8a; }
        .result { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 4px; white-space: pre-wrap; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .stats { background: #e3f2fd; padding: 15px; margin: 20px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>🔥 Simplex Fire Alarm RAG System</h1>
    
    <div class="stats">
        <h3>📊 System Status</h3>
        <p><strong>Database Components:</strong> <span id="component-count">Loading...</span></p>
        <p><strong>Graph Relationships:</strong> <span id="relationship-count">Loading...</span></p>
        <p><strong>System Status:</strong> <span id="system-status">{{ system_status }}</span></p>
    </div>

    <div class="container">
        <div class="panel">
            <h2>❓ Ask Questions</h2>
            <p>Ask technical questions about Simplex fire alarm systems:</p>
            <textarea id="question" placeholder="Example: What fire alarm control panels support IDNet protocol?">What are the main Simplex fire alarm control panels?</textarea>
            <button onclick="askQuestion()">Ask Question</button>
            <div id="answer-result" class="result" style="display:none;"></div>
        </div>

        <div class="panel">
            <h2>📋 Process BOQ</h2>
            <p>Enter bill of quantities for system sizing:</p>
            <textarea id="boq" placeholder="Example: Office building with 50 smoke detectors, 10 manual stations, 20 horn strobes">Fire alarm system for 50-room hotel: 60 smoke detectors, 15 heat detectors, 10 manual stations, 25 horn strobes</textarea>
            <button onclick="processBOQ()">Process BOQ</button>
            <div id="boq-result" class="result" style="display:none;"></div>
        </div>
    </div>

    <script>
        // Load system stats on page load
        window.onload = function() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('component-count').textContent = data.components;
                    document.getElementById('relationship-count').textContent = data.relationships;
                })
                .catch(error => {
                    document.getElementById('component-count').textContent = 'Error loading';
                    document.getElementById('relationship-count').textContent = 'Error loading';
                });
        };

        async function askQuestion() {
            const question = document.getElementById('question').value;
            const resultDiv = document.getElementById('answer-result');
            
            if (!question.trim()) {
                alert('Please enter a question');
                return;
            }

            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '🔄 Processing question...';

            try {
                const response = await fetch('/api/question', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });

                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = `<strong>Answer:</strong>\\n${data.answer}\\n\\n<strong>Context:</strong> Found ${data.context_count} relevant components`;
                } else {
                    resultDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
            }
        }

        async function processBOQ() {
            const boq = document.getElementById('boq').value;
            const resultDiv = document.getElementById('boq-result');
            
            if (!boq.trim()) {
                alert('Please enter BOQ requirements');
                return;
            }

            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '🔄 Processing BOQ...';

            try {
                const response = await fetch('/api/boq', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({boq: boq})
                });

                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = `<strong>System Configuration:</strong>\\n${data.component_count} components selected\\n\\n<strong>Validation:</strong> ${data.valid ? 'PASSED' : 'FAILED'}\\n\\n<strong>Report:</strong>\\n${data.report.substring(0, 500)}...`;
                } else {
                    resultDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Main web interface."""
    system_status = "✅ Online" if orch else "❌ Offline"
    return render_template_string(HTML_TEMPLATE, system_status=system_status)

@app.route('/api/stats')
def get_stats():
    """Get system statistics with fresh database connection."""
    try:
        # Create a fresh database connection to get current stats using absolute path
        from simplex_rag.database import SimplexDatabase
        fresh_db = SimplexDatabase(db_path='/root/simplex_components.db')
        
        components = len(fresh_db.graph.nodes())
        relationships = len(fresh_db.graph.edges())
        
        return jsonify({
            "components": components,
            "relationships": relationships,
            "status": "online"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/question', methods=['POST'])
def answer_question():
    """Answer a technical question with fresh database connection."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({"success": False, "error": "No question provided"}), 400
        
        # Create fresh orchestrator with correct database path
        from simplex_rag.orchestrator import SimplexRAGOrchestrator
        from simplex_rag.database import SimplexDatabase
        
        fresh_db = SimplexDatabase(db_path='/root/simplex_components.db')
        fresh_orch = SimplexRAGOrchestrator()
        fresh_orch.db = fresh_db  # Use the fresh database
        
        answer, contexts = fresh_orch.answer_question(question)
        
        return jsonify({
            "success": True,
            "answer": answer,
            "context_count": len(contexts)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/boq', methods=['POST'])
def process_boq():
    """Process a Bill of Quantities with fresh database connection."""
    try:
        data = request.get_json()
        boq_text = data.get('boq', '')
        
        if not boq_text:
            return jsonify({"success": False, "error": "No BOQ provided"}), 400
        
        # Create fresh orchestrator with correct database path
        from simplex_rag.orchestrator import SimplexRAGOrchestrator
        from simplex_rag.database import SimplexDatabase
        
        fresh_db = SimplexDatabase(db_path='/root/simplex_components.db')
        fresh_orch = SimplexRAGOrchestrator()
        fresh_orch.db = fresh_db  # Use the fresh database
        
        config, validation, report = fresh_orch.process_boq(boq_text)
        
        return jsonify({
            "success": True,
            "component_count": len(config.get('components', [])),
            "valid": validation.get('valid', False),
            "report": report
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Simplex RAG Web Interface...")
    print("📍 Access at: http://localhost:8081")
    print("🔧 System ready for fire alarm queries and BOQ processing")
    
    app.run(host='0.0.0.0', port=8081, debug=False)