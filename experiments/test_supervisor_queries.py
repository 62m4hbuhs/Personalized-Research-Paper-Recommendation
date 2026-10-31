from backend.hybrid_recommender import HybridRecommender

rec = HybridRecommender()
test_titles = [
    'Attention is All you Need',
    'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
    'Matrix Factorization Techniques for Recommender Systems',
    'A Simple Framework for Contrastive Learning of Visual Representations'
]

for title in test_titles:
    pid, meta = rec.get_paper_by_title_or_id(title)
    print(f"\n==========================================")
    print(f"SEED: {title}")
    print(f"==========================================")
    if not pid:
        print("Not found")
        continue
    res = rec.recommend(seed_paper_id=pid, top_k=3, alpha=0.6)
    for r in res['recommendations']:
        print(f"#{r['rank']} [Score: {r['hybrid_score']:.3f} | Sem: {r['semantic_score']:.1%} | Graph: {r['graph_score']:.1%}]")
        print(f"   {r['title']} ({r['year']}) | Citations: {r['citationCount']:,}")
        print(f"   Why: {r['explanation']}")

