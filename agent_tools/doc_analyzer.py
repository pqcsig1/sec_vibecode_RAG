#!/usr/bin/env python3
"""
Document analysis tool for secure vibe coding RAG system
Provides document metadata, statistics, and content analysis.

Security notes:
- Does not execute untrusted code; reads filesystem metadata only.
- DATA_DIR is resolved from environment or local project to avoid hardcoded paths.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import hashlib

def document_analyzer(query: str) -> str:
    """
    Analyze documents in the knowledge base
    Provides metadata, statistics, and content insights
    """
    try:
        # Resolve data directory securely: prefer env var, else common local paths
        data_dir = os.getenv("DATA_DIR")
        if not data_dir:
            candidates = [
                os.path.join(os.getcwd(), "data"),
                str(Path(__file__).resolve().parents[1] / "data"),
            ]
            for p in candidates:
                if os.path.isdir(p):
                    data_dir = p
                    break
        if not data_dir or not os.path.isdir(data_dir):
            return "Error: Data directory not found"
        
        analysis = {
            "total_documents": 0,
            "document_types": {},
            "total_size_bytes": 0,
            "documents": [],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Analyze each document
        for file_path in Path(data_dir).rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    file_ext = file_path.suffix.lower()
                    
                    # Calculate file hash for integrity
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                    
                    doc_info = {
                        "filename": file_path.name,
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": file_ext,
                        "hash": file_hash
                    }
                    
                    analysis["documents"].append(doc_info)
                    analysis["total_documents"] += 1
                    analysis["total_size_bytes"] += stat.st_size
                    
                    # Count by type
                    if file_ext in analysis["document_types"]:
                        analysis["document_types"][file_ext] += 1
                    else:
                        analysis["document_types"][file_ext] = 1
                        
                except Exception as e:
                    continue
        
        # Format response based on query
        if "count" in query.lower() or "how many" in query.lower():
            return f"Knowledge base contains {analysis['total_documents']} documents"
        elif "size" in query.lower():
            size_mb = analysis['total_size_bytes'] / (1024 * 1024)
            return f"Total knowledge base size: {size_mb:.2f} MB ({analysis['total_size_bytes']} bytes)"
        elif "types" in query.lower() or "format" in query.lower():
            types_str = ", ".join([f"{ext}: {count}" for ext, count in analysis['document_types'].items()])
            return f"Document types: {types_str}"
        else:
            # Return comprehensive analysis
            return json.dumps(analysis, indent=2)
            
    except Exception as e:
        return f"Error analyzing documents: {str(e)}"
