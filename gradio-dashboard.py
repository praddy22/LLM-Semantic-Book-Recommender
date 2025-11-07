import pandas as pd
import numpy as np
from dotenv import load_dotenv
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
book_vectors = vectorizer.fit_transform(books['description'].fillna(''))


def decompose_query_intents(query: str):
    """Decompose a single query into distinct intents using pattern matching and keywords"""
    intents = {
        'mood': [],
        'pacing': [],
        'trope': [],
        'theme': []
    }
    
    # Define keyword patterns for different intents
    mood_keywords = {
        'happy': ['happy', 'joyful', 'cheerful', 'uplifting', 'positive', 'funny', 'humorous'],
        'sad': ['sad', 'melancholy', 'depressing', 'emotional', 'heartbreaking', 'tear-jerking'],
        'angry': ['angry', 'rage', 'furious', 'frustrating', 'intense', 'passionate'],
        'fearful': ['scary', 'terrifying', 'horror', 'suspense', 'thrilling', 'chilling'],
        'surprised': ['surprising', 'unexpected', 'twisty', 'shocking', 'mind-bending']
    }
    
    pacing_keywords = {
        'slow': ['slow', 'leisurely', 'relaxed', 'contemplative', 'meditative', 'gradual'],
        'fast': ['fast', 'paced', 'action-packed', 'quick', 'racing', 'intense'],
        'steady': ['steady', 'balanced', 'measured', 'consistent', 'even']
    }
    
    trope_keywords = {
        'romance': ['romance', 'love story', 'romantic', 'relationship', 'couple'],
        'mystery': ['mystery', 'detective', 'investigation', 'clue', 'puzzle'],
        'coming_of_age': ['coming of age', 'growing up', 'adolescence', 'maturity', 'youth'],
        'quest': ['quest', 'journey', 'adventure', 'hero', 'mission'],
        'revenge': ['revenge', 'vengeance', 'retribution', 'payback']
    }
    
    theme_keywords = {
        'forgiveness': ['forgiveness', 'redemption', 'second chance', 'atonement'],
        'identity': ['identity', 'self-discovery', 'who am i', 'belonging'],
        'justice': ['justice', 'fairness', 'equality', 'rights', 'law'],
        'freedom': ['freedom', 'liberty', 'independence', 'escape'],
        'family': ['family', 'parent', 'child', 'relationship', 'bond']
    }
    
    query_lower = query.lower()
    
    # Extract intents
    for mood, keywords in mood_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            intents['mood'].append(mood)
    
    for pacing, keywords in pacing_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            intents['pacing'].append(pacing)
    
    for trope, keywords in trope_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            intents['trope'].append(trope)
    
    for theme, keywords in theme_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            intents['theme'].append(theme)
    
    return intents


def apply_quality_safety_reranking(recommendations: pd.DataFrame, min_rating: float = 3.5, max_age_rating: str = "adult") -> pd.DataFrame:
    """Apply constraint-aware quality and safety re-ranking"""
    
    # Define age rating hierarchy (lower numbers = more restrictive)
    age_hierarchy = {
        'children': 1,
        'young_adult': 2, 
        'adult': 3
    }
    
    max_age_level = age_hierarchy.get(max_age_rating, 3)
    
    # Filter by quality metrics
    filtered_recs = recommendations.copy()
    
    # Apply rating filter if available
    if 'average_rating' in filtered_recs.columns:
        filtered_recs = filtered_recs[filtered_recs['average_rating'] >= min_rating]
    
    # Apply age-appropriate filtering (simplified - in real implementation would use actual age ratings)
    # For demo purposes, we'll assume books with certain keywords are age-appropriate
    adult_keywords = ['violence', 'sex', 'drugs', 'profanity', 'mature', 'adult']
    young_adult_keywords = ['teen', 'young adult', 'coming of age', 'school']
    
    def get_age_rating(description):
        desc_lower = description.lower() if isinstance(description, str) else ""
        if any(keyword in desc_lower for keyword in adult_keywords):
            return 'adult'
        elif any(keyword in desc_lower for keyword in young_adult_keywords):
            return 'young_adult'
        else:
            return 'children'  # Default to most restrictive
    
    filtered_recs['inferred_age_rating'] = filtered_recs['description'].apply(get_age_rating)
    filtered_recs['age_level'] = filtered_recs['inferred_age_rating'].map(age_hierarchy)
    filtered_recs = filtered_recs[filtered_recs['age_level'] <= max_age_level]
    
    # Calculate composite quality score
    filtered_recs['quality_score'] = (
        filtered_recs.get('average_rating', 0) * 0.4 +  # 40% weight on user ratings
        filtered_recs.get('num_pages', 0).clip(0, 500) / 500 * 0.3 +  # 30% weight on length (normalized)
        (filtered_recs.get('ratings_count', 0) / filtered_recs.get('ratings_count', 1).max()) * 0.3  # 30% weight on popularity
    )
    
    # Re-rank by combining semantic similarity with quality score
    # This is a simplified constrained optimization
    filtered_recs['final_score'] = (
        filtered_recs.get('semantic_similarity', 0.5) * 0.7 +  # 70% semantic relevance
        filtered_recs['quality_score'] * 0.3  # 30% quality
    )
    
    filtered_recs = filtered_recs.sort_values('final_score', ascending=False)
    
    return filtered_recs


def retrieve_multi_intent_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
        quality_filter: bool = True,
        min_rating: float = 3.5,
        age_rating: str = "adult"
) -> tuple:
    """Enhanced recommendation with multi-intent fusion and quality/safety constraints"""
    
    # Decompose query into intents
    intents = decompose_query_intents(query)
    
    # Get base semantic recommendations
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, book_vectors).flatten()
    
    # Get top book indices
    top_indices = similarities.argsort()[-initial_top_k:][::-1]
    book_recs = books.iloc[top_indices].copy()
    
    # Add semantic similarity score for later use
    book_recs['semantic_similarity'] = similarities[top_indices]
    
    # Apply category filter
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)
    
    # Enhanced tone filtering based on detected mood intents
    if tone == "All" and intents['mood']:
        # Use detected mood to influence sorting
        mood_scores = {}
        for mood in intents['mood']:
            if mood == 'happy':
                mood_scores['joy'] = book_recs.get('joy', 0)
            elif mood == 'sad':
                mood_scores['sadness'] = book_recs.get('sadness', 0)
            elif mood == 'angry':
                mood_scores['anger'] = book_recs.get('anger', 0)
            elif mood == 'fearful':
                mood_scores['fear'] = book_recs.get('fear', 0)
            elif mood == 'surprised':
                mood_scores['surprise'] = book_recs.get('surprise', 0)
        
        if mood_scores:
            # Create composite mood score
            book_recs['mood_score'] = sum(mood_scores.values()) / len(mood_scores)
            book_recs.sort_values(by="mood_score", ascending=False, inplace=True)
    else:
        # Original tone filtering
        if tone == "Happy":
            book_recs.sort_values(by="joy", ascending=False, inplace=True)
        elif tone == "Surprising":
            book_recs.sort_values(by="surprise", ascending=False, inplace=True)
        elif tone == "Angry":
            book_recs.sort_values(by="anger", ascending=False, inplace=True)
        elif tone == "Suspenseful":
            book_recs.sort_values(by="fear", ascending=False, inplace=True)
        elif tone == "Sad":
            book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    
    # Apply quality and safety re-ranking if enabled
    if quality_filter:
        book_recs = apply_quality_safety_reranking(book_recs, min_rating, age_rating)
    
    return book_recs, intents


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    # Get the vector for the query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, book_vectors).flatten()
    
    # Get top book indices
    top_indices = similarities.argsort()[-initial_top_k:][::-1]
    book_recs = books.iloc[top_indices].copy()

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def generate_counterfactual_explanations(query: str, category: str, tone: str, top_book_title: str):
    """Generate counterfactual explanations by ablating different factors"""
    explanations = []
    
    # Get original recommendations
    original_recs = retrieve_semantic_recommendations(query, category, tone, final_top_k=50)
    original_rank = None
    for idx, row in original_recs.iterrows():
        if row['title'] == top_book_title:
            original_rank = idx + 1
            break
    
    if original_rank is None:
        return ["Book not found in recommendations"]
    
    # Ablate category filter
    if category != "All":
        no_category_recs = retrieve_semantic_recommendations(query, "All", tone, final_top_k=50)
        no_category_rank = None
        for idx, row in no_category_recs.iterrows():
            if row['title'] == top_book_title:
                no_category_rank = idx + 1
                break
        
        if no_category_rank:
            rank_change = original_rank - no_category_rank
            if rank_change > 0:
                explanations.append(f"📈 Rank improved by {rank_change} positions when category filter removed")
            elif rank_change < 0:
                explanations.append(f"📉 Rank dropped by {abs(rank_change)} positions when category filter removed")
            else:
                explanations.append("🎯 Category filter had minimal impact on ranking")
    
    # Ablate tone filter
    if tone != "All":
        no_tone_recs = retrieve_semantic_recommendations(query, category, "All", final_top_k=50)
        no_tone_rank = None
        for idx, row in no_tone_recs.iterrows():
            if row['title'] == top_book_title:
                no_tone_rank = idx + 1
                break
        
        if no_tone_rank:
            rank_change = original_rank - no_tone_rank
            if rank_change > 0:
                explanations.append(f"📈 Rank improved by {rank_change} positions when tone filter removed")
            elif rank_change < 0:
                explanations.append(f"📉 Rank dropped by {abs(rank_change)} positions when tone filter removed")
            else:
                explanations.append("🎯 Tone filter had minimal impact on ranking")
    
    # Semantic similarity analysis
    query_vector = vectorizer.transform([query])
    book_idx = books[books['title'] == top_book_title].index[0]
    similarity_score = cosine_similarity(query_vector, book_vectors[book_idx:book_idx+1])[0][0]
    explanations.append(f"🔍 Semantic similarity score: {similarity_score:.3f}")
    
    # Emotional tone analysis
    book_emotions = books.loc[book_idx, ['joy', 'sadness', 'surprise', 'anger', 'fear', 'disgust', 'neutral']]
    top_emotion = book_emotions.idxmax()
    top_score = book_emotions.max()
    explanations.append(f"🎭 Dominant emotion: {top_emotion} ({top_score:.3f})")
    
    return explanations


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = str(row["description"]) if pd.notna(row["description"]) else "No description available"
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = str(row["authors"]).split(";") if pd.notna(row["authors"]) else ["Unknown Author"]
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"] if pd.notna(row["authors"]) else "Unknown Author"

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


def recommend_books_with_explanations(
        query: str,
        category: str,
        tone: str,
        quality_filter: bool,
        min_rating: float,
        age_rating: str
):
    # Use multi-intent recommendations with quality/safety constraints
    recommendations, intents = retrieve_multi_intent_recommendations(
        query, category, tone, quality_filter=quality_filter, 
        min_rating=min_rating, age_rating=age_rating
    )
    
    results = []
    explanations = []
    
    # Format detected intents for display
    intent_summary = []
    for intent_type, detected_intents in intents.items():
        if detected_intents:
            intent_summary.append(f"🎯 {intent_type.title()}: {', '.join(detected_intents)}")
    
    intent_text = "📋 Detected Intents:\n" + "\n".join(intent_summary) if intent_summary else "📋 No specific intents detected - using general semantic matching"
    
    # Add quality/safety information
    quality_info = f"\n\n🛡️ Quality & Safety Filters: {'Enabled' if quality_filter else 'Disabled'}"
    if quality_filter:
        quality_info += f"\n   • Minimum rating: {min_rating}⭐"
        quality_info += f"\n   • Age appropriateness: {age_rating.replace('_', ' ').title()}"
        quality_info += f"\n   • Books shown: {len(recommendations)} (after filtering)"

    for _, row in recommendations.iterrows():
        description = str(row["description"]) if pd.notna(row["description"]) else "No description available"
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = str(row["authors"]).split(";") if pd.notna(row["authors"]) else ["Unknown Author"]
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"] if pd.notna(row["authors"]) else "Unknown Author"

        # Add quality indicators to caption
        rating_info = f" ({row.get('average_rating', 'N/A')}⭐)" if 'average_rating' in row else ""
        age_info = f" [{row.get('inferred_age_rating', 'unknown').replace('_', ' ')}]" if 'inferred_age_rating' in row else ""
        
        caption = f"{row['title']} by {authors_str}{rating_info}{age_info}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
        
        # Generate explanations for the first (top) recommendation
        if len(explanations) == 0:
            explanations = generate_counterfactual_explanations(query, category, tone, row['title'])
    
    full_explanation = intent_text + quality_info + "\n\n" + "🔍 Counterfactual Analysis:\n" + "\n".join(explanations)
    return results, full_explanation

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

css = """
footer {display: none !important;}
.gradio-container {max-width: 1200px !important;}
"""

with gr.Blocks(theme = gr.themes.Glass(), css=css) as dashboard:
    gr.Markdown("# 📚 Advanced Semantic Book Recommender")
    gr.Markdown("✨ **New Features:** Multi-intent semantic fusion, counterfactual explanations, and quality/safety filtering")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A slow-paced mystery about forgiveness and redemption")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")

    with gr.Row():
        quality_checkbox = gr.Checkbox(label="Enable quality & safety filtering", value=True)
        min_rating_slider = gr.Slider(minimum=1.0, maximum=5.0, value=3.5, step=0.1, label="Minimum rating")
        age_rating_dropdown = gr.Dropdown(choices=["children", "young_adult", "adult"], 
                                         label="Age appropriateness", value="adult")
        submit_button = gr.Button("🔍 Find recommendations")

    gr.Markdown("## 📖 Recommendations")
    output = gr.Gallery(label = "Recommended books (with quality indicators)", columns = 8, rows = 2)
    
    gr.Markdown("## 🧠 AI Analysis & Explanations")
    explanations_output = gr.Textbox(
        label="Multi-intent analysis, quality metrics, and counterfactual explanations",
        lines=15,
        interactive=False,
        placeholder="Click 'Find recommendations' to see AI-powered analysis..."
    )

    submit_button.click(fn = recommend_books_with_explanations,
                        inputs = [user_query, category_dropdown, tone_dropdown, 
                                quality_checkbox, min_rating_slider, age_rating_dropdown],
                        outputs = [output, explanations_output])


if __name__ == "__main__":
    dashboard.launch(share=True)  # This will create a public URL