from typing import List, Dict, Any, Set

def extract_called_identifiers(node, content: str) -> List[str]:
    """
    Recursively extract potential called function/method identifiers from a tree-sitter node.
    This is a best-effort approach that looks for 'call' nodes and extracts the identifier.
    """
    callees = set()
    _find_calls(node, content, callees)
    return list(callees)

def _find_calls(node, content: str, callees: Set[str]):
    node_type = node.type.lower()
    
    # Check if this node represents a call
    if "call" in node_type:
        # The first child or an identifier child is typically the function being called
        # We look for children that are identifiers, or traverse down to an identifier
        for child in node.children:
            if "identifier" in child.type.lower() or child.type == "name" or child.type == "property_identifier":
                try:
                    callee_name = content[child.start_byte:child.end_byte]
                    callees.add(callee_name)
                except Exception:
                    pass
            # Sometimes the call expression contains an attribute/member expression
            # e.g., obj.method(). The child might be a 'member_expression' or 'attribute'
            elif "attribute" in child.type.lower() or "member" in child.type.lower() or "selector" in child.type.lower():
                _extract_from_member(child, content, callees)

    # Continue traversing to find nested calls
    for child in node.children:
        _find_calls(child, content, callees)

def _extract_from_member(node, content: str, callees: Set[str]):
    """
    Extract method name from expressions like obj.method()
    """
    # Usually the last child or property_identifier holds the method name
    for child in node.children:
        if "identifier" in child.type.lower() or child.type == "name" or child.type == "property_identifier":
            try:
                callee_name = content[child.start_byte:child.end_byte]
                callees.add(callee_name)
            except Exception:
                pass
        else:
            _extract_from_member(child, content, callees)

def build_call_graph(chunks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build a simple adjacency list representing the call graph across all chunks.
    Nodes are symbol_names, edges are calls to other known symbol_names.
    """
    # 1. Collect all known symbol names
    known_symbols = set()
    for chunk in chunks:
        if "symbol_name" in chunk["metadata"] and chunk["metadata"]["symbol_name"] != "unknown":
            known_symbols.add(chunk["metadata"]["symbol_name"])
            
            # also add the raw name if it's a class method (e.g., MyClass.my_method -> my_method)
            symbol_parts = chunk["metadata"]["symbol_name"].split(".")
            if len(symbol_parts) > 1:
                 known_symbols.add(symbol_parts[-1])
            
    # 2. Filter the callees for each chunk to only include those in known_symbols
    call_graph = {}
    
    for chunk in chunks:
        caller = chunk["metadata"].get("symbol_name")
        callees = chunk["metadata"].get("callees", [])
        
        if caller and caller != "unknown":
            valid_callees = []
            for callee in callees:
                if callee in known_symbols and callee != caller:
                    # Try to map back to the full symbol name if possible, or just use the extracted name
                    valid_callees.append(callee)
            
            call_graph[caller] = list(set(valid_callees))
            
    return call_graph
