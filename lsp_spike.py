import subprocess
import json
import threading
from pathlib import Path
from pylsp_jsonrpc.streams import JsonRpcStreamWriter, JsonRpcStreamReader
from dotenv import load_dotenv
import os
import time

load_dotenv()

LSP_CMD = os.getenv('LSP_COMMAND', 'pylsp').split()
LSP_TIMEOUT = int(os.getenv('LSP_TIMEOUT', '15'))
LSP_DIAGNOSTIC_DELAY = float(os.getenv('LSP_DIAGNOSTIC_DELAY', '0.5'))

class LspClient:
    def __init__(self, repo_path: Path):
        self.repo_path_uri = repo_path.as_uri()
        self.process = None
        self.writer = None
        self.reader = None
        self.reader_thread = None
        self.message_id_counter = 1
        self.responses = {}
        self.notifications = []
        self._lock = threading.Lock()
        self._response_events = {}
        self._diagnostics = {}

    def start(self):
        print("LSP: Starting server...")
        self.process = subprocess.Popen(
            LSP_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.writer = JsonRpcStreamWriter(self.process.stdin)
        self.reader = JsonRpcStreamReader(self.process.stdout)
        
        self.reader_thread = threading.Thread(target=self._reader_loop)
        self.reader_thread.daemon = True
        self.reader_thread.start()
        print("LSP: Server started.")

    def _consume_message(self, msg):
        if 'id' in msg:
            with self._lock:
                self.responses[msg['id']] = msg
                if msg['id'] in self._response_events:
                    self._response_events[msg['id']].set()
        else:
            if msg.get('method') == 'textDocument/publishDiagnostics':
                params = msg.get('params', {})
                uri = params.get('uri')
                if uri:
                    self._diagnostics[uri] = params.get('diagnostics', [])
            self.notifications.append(msg)

    def _reader_loop(self):
        self.reader.listen(self._consume_message)

    def send_request(self, method, params):
        msg_id = self.message_id_counter
        self.message_id_counter += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params,
        }
        
        event = threading.Event()
        with self._lock:
            self._response_events[msg_id] = event
        
        self.writer.write(request)
        event.wait(timeout=LSP_TIMEOUT)
        
        with self._lock:
            return self.responses.pop(msg_id, None)

    def initialize(self):
        print("LSP: Initializing project...")
        response = self.send_request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.repo_path_uri,
                "capabilities": {
                    "textDocument": {
                        "hover": {"contentFormat": ["plaintext", "markdown"]},
                        "definition": {"dynamicRegistration": True},
                        "references": {"dynamicRegistration": True},
                        "documentHighlight": {"dynamicRegistration": True},
                        "documentSymbol": {
                            "dynamicRegistration": True,
                            "hierarchicalDocumentSymbolSupport": True
                        },
                        "codeAction": {"dynamicRegistration": True},
                        "completion": {
                            "dynamicRegistration": True,
                            "completionItem": {"snippetSupport": True}
                        },
                        "signatureHelp": {"dynamicRegistration": True},
                        "publishDiagnostics": {"relatedInformation": True},
                        "implementation": {"dynamicRegistration": True},
                        "typeDefinition": {"dynamicRegistration": True}
                    },
                    "workspace": {
                        "symbol": {"dynamicRegistration": True},
                        "executeCommand": {"dynamicRegistration": True}
                    }
                },
            },
        )
        self.writer.write({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        print("LSP: Project initialized.")
        return response

    def _ensure_absolute_uri(self, file_path: str) -> str:
        absolute_file_path = Path(file_path).resolve()
        return absolute_file_path.as_uri()

    def open_file(self, file_path: str):
        file_uri = self._ensure_absolute_uri(file_path)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
        
        self.writer.write({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": file_uri,
                    "languageId": "python",
                    "version": 1,
                    "text": content
                }
            }
        })
        time.sleep(LSP_DIAGNOSTIC_DELAY)
        
    def get_diagnostics(self, file_path: str):
        file_uri = self._ensure_absolute_uri(file_path)
        self.open_file(file_path)
        return self._diagnostics.get(file_uri, [])

    def get_references(self, file_path: str, line: int, character: int):
        print(f"LSP: Getting references for {file_path}:{line}:{character}")
        file_uri = self._ensure_absolute_uri(file_path)
        
        response = self.send_request(
            "textDocument/references",
            {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": False},
            },
        )
        return response

    def get_definition(self, file_path: str, line: int, character: int):
        file_uri = self._ensure_absolute_uri(file_path)
        
        response = self.send_request(
            "textDocument/definition",
            {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": character}
            }
        )
        return response

    def get_implementations(self, file_path: str, line: int, character: int):
        file_uri = self._ensure_absolute_uri(file_path)
        
        response = self.send_request(
            "textDocument/implementation",
            {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": character}
            }
        )
        return response

    def get_type_definition(self, file_path: str, line: int, character: int):
        file_uri = self._ensure_absolute_uri(file_path)
        
        response = self.send_request(
            "textDocument/typeDefinition",
            {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": character}
            }
        )
        return response

    def get_hover(self, file_path: str, line: int, character: int):
        file_uri = self._ensure_absolute_uri(file_path)
        
        response = self.send_request(
            "textDocument/hover",
            {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": character}
            }
        )
        return response

    def get_document_symbols(self, file_path: str):
        file_uri = self._ensure_absolute_uri(file_path)
        self.open_file(file_path)
        
        response = self.send_request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": file_uri}}
        )
        return response

    def get_workspace_symbols(self, query: str):
        response = self.send_request(
            "workspace/symbol",
            {"query": query}
        )
        return response

    def get_signature_help(self, file_path: str, line: int, character: int):
        file_uri = self._ensure_absolute_uri(file_path)
        
        response = self.send_request(
            "textDocument/signatureHelp",
            {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": character}
            }
        )
        return response

    def get_code_actions(self, file_path: str, start_line: int, start_char: int, 
                        end_line: int, end_char: int, diagnostics=None):
        file_uri = self._ensure_absolute_uri(file_path)
        
        params = {
            "textDocument": {"uri": file_uri},
            "range": {
                "start": {"line": start_line, "character": start_char},
                "end": {"line": end_line, "character": end_char}
            },
            "context": {
                "diagnostics": diagnostics or []
            }
        }
        
        response = self.send_request("textDocument/codeAction", params)
        return response

    def get_incoming_calls(self, file_path: str, line: int, character: int):
        file_uri = self._ensure_absolute_uri(file_path)
        
        prepare_response = self.send_request(
            "textDocument/prepareCallHierarchy",
            {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": character}
            }
        )
        
        if not prepare_response or 'result' not in prepare_response:
            return None
        
        items = prepare_response.get('result', [])
        if not items:
            return None
        
        response = self.send_request(
            "callHierarchy/incomingCalls",
            {"item": items[0]}
        )
        return response

    def get_outgoing_calls(self, file_path: str, line: int, character: int):
        file_uri = self._ensure_absolute_uri(file_path)
        
        prepare_response = self.send_request(
            "textDocument/prepareCallHierarchy",
            {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": character}
            }
        )
        
        if not prepare_response or 'result' not in prepare_response:
            return None
        
        items = prepare_response.get('result', [])
        if not items:
            return None
        
        response = self.send_request(
            "callHierarchy/outgoingCalls",
            {"item": items[0]}
        )
        return response

    def shutdown(self):
        print("LSP: Shutting down server...")
        self.send_request("shutdown", None)
        self.writer.write({"jsonrpc": "2.0", "method": "exit", "params": None})
        
        if self.process:
            self.process.terminate()
        if self.reader_thread:
            self.reader_thread.join(timeout=2)
        print("LSP: Server shut down.")
