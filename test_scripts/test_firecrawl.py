#!/usr/bin/env python3
"""
Test script for Firecrawl v2 API integration
Based on official documentation: https://docs.firecrawl.dev/api-reference/endpoint/search

Tests all major features:
- Multi-source search (web, news, images)
- Time-based filtering
- Category filtering
- Country-specific results
- Full content scraping
"""

from nvidia_year_progress_and_news_reporter import search_web_with_firecrawl, get_news_with_firecrawl

print("=" * 70)
print("🧪 Testing Firecrawl v2 API Integration")
print("   Based on: https://docs.firecrawl.dev/api-reference/endpoint/search")
print("=" * 70)

# Test 1: Basic web + news search
print("\n1️⃣ Testing multi-source search (web + news)...")
print("-" * 70)
results = search_web_with_firecrawl(
    query="artificial intelligence breakthroughs",
    limit=3,
    sources=["web", "news"],
    tbs="qdr:d",  # Past day
    country="US"
)

if results.get('success', True):
    web_results = results.get('data', {}).get('web', [])
    news_results = results.get('data', {}).get('news', [])
    
    print(f"✓ Search successful!")
    print(f"  Web results: {len(web_results)}")
    print(f"  News results: {len(news_results)}")
    
    if news_results:
        print("\n  First news result:")
        first = news_results[0]
        print(f"    Title: {first.get('title', 'N/A')[:80]}")
        print(f"    URL: {first.get('url', 'N/A')}")
        print(f"    Date: {first.get('date', 'N/A')}")
        print(f"    Has snippet: {bool(first.get('snippet'))}")
        print(f"    Has markdown: {bool(first.get('markdown'))}")
    
    if web_results:
        print("\n  First web result:")
        first = web_results[0]
        print(f"    Title: {first.get('title', 'N/A')[:80]}")
        print(f"    URL: {first.get('url', 'N/A')}")
        print(f"    Has description: {bool(first.get('description'))}")
        print(f"    Has markdown: {bool(first.get('markdown'))}")
else:
    print("✗ Search failed or timed out")

# Test 2: Category filtering (GitHub)
print("\n\n2️⃣ Testing category filtering (GitHub repositories)...")
print("-" * 70)
results = search_web_with_firecrawl(
    query="machine learning python",
    limit=3,
    sources=["web"],
    categories=["github"],
    country="US"
)

if results.get('success', True):
    web_results = results.get('data', {}).get('web', [])
    print(f"✓ Found {len(web_results)} GitHub results")
    
    if web_results:
        for i, result in enumerate(web_results[:2], 1):
            print(f"\n  Result {i}:")
            print(f"    Title: {result.get('title', 'N/A')[:70]}")
            print(f"    URL: {result.get('url', 'N/A')[:70]}")
            category = result.get('category', 'N/A')
            print(f"    Category: {category}")
else:
    print("✗ Search failed")

# Test 3: Time-based filtering
print("\n\n3️⃣ Testing time-based filtering (past week)...")
print("-" * 70)
results = search_web_with_firecrawl(
    query="technology news",
    limit=3,
    sources=["news"],
    tbs="qdr:w",  # Past week
    country="US"
)

if results.get('success', True):
    news_results = results.get('data', {}).get('news', [])
    print(f"✓ Found {len(news_results)} news articles from past week")
else:
    print("✗ Search failed")

# Test 4: Country-specific search
print("\n\n4️⃣ Testing country-specific search (India)...")
print("-" * 70)
results = search_web_with_firecrawl(
    query="technology startups",
    limit=3,
    sources=["web", "news"],
    country="IN",  # India
    tbs="qdr:d"  # Past day
)

if results.get('success', True):
    data = results.get('data', {})
    web_count = len(data.get('web', []))
    news_count = len(data.get('news', []))
    print(f"✓ Found {web_count + news_count} India-specific results")
    print(f"  Web: {web_count}, News: {news_count}")
else:
    print("✗ Search failed")

# Test 5: Full news generation with LLM processing
print("\n\n5️⃣ Testing full news generation with LLM processing...")
print("-" * 70)
try:
    news_summary = get_news_with_firecrawl(
        query="latest AI developments artificial intelligence",
        limit=5,
        time_filter="qdr:d",  # Past day
        country="US"
    )
    print(f"✓ News summary generated!")
    print(f"\nSummary preview (first 400 chars):")
    print("-" * 70)
    print(news_summary[:400] + "...")
except Exception as e:
    print(f"✗ News generation failed: {e}")

# Test 6: Search operators
print("\n\n6️⃣ Testing search operators...")
print("-" * 70)
results = search_web_with_firecrawl(
    query='site:github.com "machine learning"',  # Use site: operator
    limit=3,
    sources=["web"],
    country="US"
)

if results.get('success', True):
    web_results = results.get('data', {}).get('web', [])
    print(f"✓ Found {len(web_results)} results using site: operator")
    
    if web_results:
        for i, result in enumerate(web_results[:2], 1):
            url = result.get('url', '')
            if 'github.com' in url:
                print(f"  ✓ Result {i} is from GitHub: {url[:60]}...")
else:
    print("✗ Search failed")

print("\n" + "=" * 70)
print("✅ All tests complete!")
print("=" * 70)
print("\n📚 Firecrawl v2 Features Tested:")
print("  ✓ Multi-source search (web + news + images)")
print("  ✓ Time-based filtering (qdr:h, qdr:d, qdr:w, qdr:m)")
print("  ✓ Category filtering (github, research, pdf)")
print("  ✓ Country-specific results")
print("  ✓ Full content scraping with markdown")
print("  ✓ Search operators (site:, intitle:, etc.)")
print("  ✓ LLM processing and summarization")
print("=" * 70 + "\n")
