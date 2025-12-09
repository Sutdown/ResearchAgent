#!/usr/bin/env python3
"""
Vector Database Integration Test Script

This script tests the vector database integration with the researcher agent.
"""

import os
import sys
from RAgents.utils.vector import VectorMemory

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_vector_memory():
    """Test vector memory functionality."""
    print("🔍 Testing Vector Memory Integration...\n")

    # Initialize vector memory
    memory = VectorMemory(persist_directory="./tests/test_vector_memory")
    print("✅ VectorMemory initialized successfully")

    # Test storing research results
    test_query = "量子计算在金融领域的应用"
    test_results = {
        "search_results": [
            {"title": "量子计算在风险管理中的应用", "content": "详细内容..."},
            {"title": "量子投资组合优化", "content": "详细内容..."}
        ]
    }

    memory.store_research_result(
        query=test_query,
        results=test_results,
        quality_score=4.5,
        metadata={"sources": ["学术", "行业报告"]}
    )
    print("✅ Research results stored successfully")

    # Test finding similar queries
    similar_queries = memory.find_similar_queries("量子计算金融应用场景", threshold=0.7)
    print(f"✅ Found {len(similar_queries)} similar queries")

    if similar_queries:
        print(f"   Most similar: {similar_queries[0]['query']}")
        print(f"   Similarity score: {similar_queries[0]['similarity']:.2f}")
    return True

def test_end_to_end():
    """Test end-to-end functionality."""
    print("\n🔄 Testing End-to-End Integration...\n")
    # Create test scenario
    memory = VectorMemory(persist_directory="./tests/test_vector_memory")

    # Store first query
    memory.store_research_result(
        query="GPT-4技术架构分析",
        results={"report": "详细的技术分析报告"},
        quality_score=4.8,
        metadata={"type": "technical_analysis"}
    )

    # Store second query
    memory.store_research_result(
        query="大模型优化策略研究",
        results={"report": "优化策略的详细报告"},
        quality_score=4.2,
        metadata={"type": "optimization_strategy"}
    )

    # Test similarity search
    similar = memory.find_similar_queries("GPT-4架构特点", threshold=0.75)
    print(f"✅ Found {len(similar)} similar queries for 'GPT-4架构特点'")

    # Test quality update
    if similar:
        memory.update_quality_score(similar[0]['query_id'], 5.0)
        print("✅ Quality score updated successfully")

    print("✅ End-to-end test passed")
    return True

def main():
    """Main test function."""
    print("🚀 Vector Database Integration Test")
    print("=" * 50)

    # Run tests
    memory_ok = test_vector_memory()
    e2e_ok = test_end_to_end()
    print("memory_ok: ", memory_ok)
    print("e2e_ok: ", e2e_ok)

if __name__ == "__main__":
    exit(main())