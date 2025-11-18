import streamlit as st
import os
import googlemaps
import google.generativeai as genai
from google.generativeai import types
from pydantic import BaseModel, Field
from typing import List, Optional
import json

# --- 1. CONFIGURATION AND API CLIENTS ---

# Set page config
st.set_page_config(
    page_title="Unseen Spots Discovery Agent",
    page_icon="🗺️",
    layout="wide"
)

# Load API keys from environment variables
try:
    MAPS_KEY = os.environ["GOOGLE_MAPS_API_KEY"]
    GEMINI_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    st.error("FATAL ERROR: Please set GOOGLE_MAPS_API_KEY and GEMINI_API_KEY environment variables.")
    st.stop()

# Initialize API Clients
try:
    gmaps = googlemaps.Client(key=MAPS_KEY)
    genai.configure(api_key=GEMINI_KEY)
    
    # Configure the Gemini model for structured output
    # Using 'gemini-1.5-flash' for speed and cost-effectiveness
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Error initializing API clients: {e}")
    st.stop()

# --- 2. PYDANTIC SCHEMAS (for Structured LLM Output) ---

class SpotVibeAnalysis(BaseModel):
    """Schema for the Vibe Analysis Agent's output."""
    vibe_adjectives: List[str] = Field(description="3-5 descriptive adjectives for the spot's atmosphere (e.g., 'cozy', 'minimalist', 'bustling').")
    unique_features: List[str] = Field(description="List of unique, non-obvious features (e.g., 'cash only', 'live jazz', 'hidden courtyard').")
    vibe_match_score: int = Field(description="A score from 1 (poor match) to 10 (perfect match) of how well this spot fits the user's requested vibe.")
    vibe_match_justification: str = Field(description="A brief explanation for the given score.")

class FinalItinerarySpot(BaseModel):
    """Schema for a single spot in the final itinerary."""
    place_name: str = Field(description="The full name of the establishment.")
    pitch_narrative: str = Field(description="A 3-4 sentence compelling, atmospheric 'pitch' for why the traveler should visit, focusing on the 'secret' or 'local' feel.")
    google_maps_link: str = Field(description="The full Google Maps URL for the location.")

class FinalItinerary(BaseModel):
    """Schema for the final list of spots."""
    itinerary_title: str = Field(description="A catchy title for this curated itinerary.")
    itinerary_spots: List[FinalItinerarySpot] = Field(description="A list of 3 curated, unseen spots.")

# --- 3. AGENT FUNCTIONS ---

@st.cache_data(ttl=3600) # Cache results for 1 hour to save API calls
def agent1_find_unseen_spots(city: str, query: str, max_reviews: int, min_rating: float, radius_m: int = 3000):
    """
    AGENT 1: TOURIST TRAP IDENTIFICATION (INVERSE FILTER)
    Finds places near the city center and applies the inverse filter.
    """
    st.write(f"🕵️ **Agent 1:** Searching for '{query}' in {city}...")
    try:
        # 1. Get city center coordinates
        geocode_result = gmaps.geocode(city)
        if not geocode_result:
            st.error(f"Could not find coordinates for city: {city}")
            return []
        
        city_center = geocode_result[0]['geometry']['location']
        
        # 2. Search for all relevant places nearby
        raw_results = gmaps.places_nearby(
            location=(city_center['lat'], city_center['lng']),
            radius=radius_m, # Search within 3km radius
            keyword=query,
            language="en"
        )
    except Exception as e:
        st.error(f"Google Maps API error during search: {e}")
        return []

    unseen_spots = []
    for place in raw_results.get('results', []):
        name = place.get('name')
        place_id = place.get('place_id')
        rating = place.get('rating', 0)
        review_count = place.get('user_ratings_total', 0)
        
        # --- 3. THE INVERSE FILTER LOGIC ---
        if rating >= min_rating and review_count <= max_reviews and review_count > 0: # Ensure it has *some* reviews
            unseen_spots.append({
                'name': name,
                'place_id': place_id,
                'rating': rating,
                'review_count': review_count,
            })
            
    st.write(f"🕵️ **Agent 1:** Found {len(unseen_spots)} potential 'unseen' spots.")
    return unseen_spots

@st.cache_data(ttl=3600)
def agent2_analyze_vibe(_unseen_spots: List[dict], user_vibe: str) -> List[dict]:
    """
    AGENT 2: VIBE ANALYSIS
    Uses Gemini to analyze reviews and score spots against the user's vibe.
    """
    st.write(f"🧠 **Agent 2:** Analyzing vibes for {len(_unseen_spots)} spots...")
    analyzed_spots = []
    
    # Configure Gemini for structured output
    json_schema = types.Schema.from_dict(SpotVibeAnalysis.model_json_schema())
    tools = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="analyze_spot_vibe",
                description="Analyzes and scores a spot based on reviews.",
                parameters=json_schema
            )
        ]
    )
    # This tool forces the model to call our function (which is just a schema)
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfig.Mode.ANY,
            allowed_function_names=["analyze_spot_vibe"]
        )
    )

    for spot in _unseen_spots:
        try:
            # 1. Get detailed reviews for the spot
            place_details = gmaps.place(
                place_id=spot['place_id'],
                fields=['reviews', 'url']
            )
            
            reviews = place_details.get('result', {}).get('reviews', [])
            spot['url'] = place_details.get('result', {}).get('url', '') # Get Maps URL
            
            if not reviews:
                st.write(f"    - Skipping {spot['name']} (no reviews found).")
                continue
            
            review_snippets = " ".join([r.get('text', '') for r in reviews])
            
            # 2. Prompt Gemini for analysis
            prompt = f"""
            You are a meticulous, native-speaking travel curator.
            Analyze the following customer reviews for a spot named '{spot['name']}'.
            The user is looking for a spot with this specific vibe: "{user_vibe}".

            Based *only* on the reviews, determine the spot's atmosphere, unique features,
            and how well it matches the user's vibe on a scale of 1-10.
            
            REVIEWS:
            ---
            {review_snippets[:3000]} 
            ---
            End of reviews. Now, provide your analysis.
            """
            
            # 3. Call Gemini with the forced function tool
            response = model.generate_content(
                prompt,
                tool_config=tool_config,
                tools=[tools],
            )
            
            # 4. Parse the structured output
            fc = response.parts[0].function_call
            if fc.name == "analyze_spot_vibe":
                args = {k: v for k, v in fc.args.items()}
                vibe_analysis = SpotVibeAnalysis.model_validate(args)
                
                # Store the structured data
                spot['vibe_analysis'] = vibe_analysis.model_dump()
                analyzed_spots.append(spot)
                st.write(f"    - Analyzed {spot['name']}. Vibe Score: {vibe_analysis.vibe_match_score}/10")
            else:
                 st.write(f"    - Skipping {spot['name']} (Gemini did not return valid analysis).")

        except Exception as e:
            st.warning(f"Error analyzing {spot['name']}: {e}")
            
    # Sort by the match score from the LLM
    analyzed_spots.sort(key=lambda x: x['vibe_analysis']['vibe_match_score'], reverse=True)
    
    st.write(f"🧠 **Agent 2:** Finished analysis. Top spot: {analyzed_spots[0]['name'] if analyzed_spots else 'None'}")
    return analyzed_spots

@st.cache_data(ttl=3600)
def agent3_generate_narrative(_top_spots: List[dict], city: str, vibe: str) -> Optional[FinalItinerary]:
    """
    AGENT 3: ITINERARY BUILDER & NARRATIVE GENERATOR
    Takes the top 3 spots and writes a compelling, journalist-style narrative.
    """
    st.write(f"✍️ **Agent 3:** Generating final narrative for {len(_top_spots)} spots...")
    if not _top_spots:
        return None

    # Prepare data for the final prompt
    spot_data_summary = []
    for spot in _top_spots:
        spot_data_summary.append({
            "name": spot['name'],
            "url": spot['url'],
            "rating": spot['rating'],
            "review_count": spot['review_count'],
            "vibe_adjectives": spot['vibe_analysis']['vibe_adjectives'],
            "unique_features": spot['vibe_analysis']['unique_features'],
            "vibe_justification": spot['vibe_analysis']['vibe_match_justification']
        })

    # Configure Gemini for structured output
    json_schema = types.Schema.from_dict(FinalItinerary.model_json_schema())
    tools = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="generate_itinerary",
                description="Creates the final itinerary with narrative pitches.",
                parameters=json_schema
            )
        ]
    )
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfig.Mode.ANY,
            allowed_function_names=["generate_itinerary"]
        )
    )

    prompt = f"""
    You are a world-class travel journalist for a luxury magazine, known for finding "secret spots".
    A traveler wants a curated list of 3 hidden gems in {city} with the vibe: "{vibe}".
    
    You have the following research data on 3 spots. Your job is to act as the final editor.
    Do NOT mention the review count or rating. Focus on atmosphere and the "unseen" quality.
    
    Create a catchy title and write a compelling, 3-4 sentence "pitch narrative" for each.
    
    RESEARCH DATA:
    ---
    {json.dumps(spot_data_summary, indent=2)}
    ---
    Now, generate the final itinerary.
    """
    
    try:
        response = model.generate_content(
            prompt,
            tool_config=tool_config,
            tools=[tools],
        )
        
        fc = response.parts[0].function_call
        if fc.name == "generate_itinerary":
            args = {k: v for k, v in fc.args.items()}
            final_itinerary = FinalItinerary.model_validate(args)
            return final_itinerary
        else:
            st.error("Agent 3 Error: LLM failed to generate final itinerary structure.")
            return None
            
    except Exception as e:
        st.error(f"Agent 3 Error during narrative generation: {e}")
        return None

# --- 4. STREAMLIT UI ---

st.title("🗺️ The 'Unseen Spots' Discovery Agent")
st.markdown("This agent pipeline finds hyper-local, crowd-free gems by filtering for **high ratings** but **low review counts**.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Define Your Trip")
    city_input = st.text_input("Target City", "Lisbon, Portugal")
    vibe_input = st.text_input("Desired Vibe / Category", "cozy cafe with vintage books")
    
    st.header("2. Configure the Inverse Filter")
    st.info("Lower 'Max Reviews' to find more "
            "exclusive, 'unseen' spots.")
            
    min_rating_input = st.slider(
        "Minimum Star Rating", 
        min_value=4.0, max_value=5.0, value=4.5, step=0.1
    )
    max_reviews_input = st.slider(
        "Maximum Review Count (Saturation Limit)", 
        min_value=50, max_value=1000, value=500, step=50
    )
    
    run_button = st.button("🚀 Find Unseen Spots", type="primary")

# --- Main Content Area ---
if run_button:
    if not city_input or not vibe_input:
        st.error("Please enter a City and a Vibe.")
    else:
        with st.spinner("Agent Pipeline is running..."):
            # Create columns for logs and results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Agent Logs")
                
                # --- RUN AGENT 1 ---
                unseen_spots = agent1_find_unseen_spots(
                    city=city_input, 
                    query=vibe_input, 
                    max_reviews=max_reviews_input, 
                    min_rating=min_rating_input
                )
                
                if unseen_spots:
                    # --- RUN AGENT 2 ---
                    analyzed_spots = agent2_analyze_vibe(
                        _unseen_spots=unseen_spots, 
                        user_vibe=vibe_input
                    )
                    
                    if analyzed_spots:
                        # --- RUN AGENT 3 ---
                        # Take only the top 3 spots for the final narrative
                        final_itinerary = agent3_generate_narrative(
                            _top_spots=analyzed_spots[:3], 
                            city=city_input, 
                            vibe=vibe_input
                        )
                        
                        # --- DISPLAY RESULTS in Column 2 ---
                        with col2:
                            st.subheader("Final Curated Itinerary")
                            if final_itinerary:
                                st.header(final_itinerary.itinerary_title)
                                st.divider()
                                
                                for spot in final_itinerary.itinerary_spots:
                                    st.subheader(spot.place_name)
                                    st.markdown(f"> _{spot.pitch_narrative}_")
                                    st.link_button("View on Google Maps", spot.google_maps_link)
                                    st.divider()
                                
                                # Show raw data for transparency
                                with st.expander("Show Agent 2's Raw Data (Top 3 Spots)"):
                                    st.json(analyzed_spots[:3])
                                    
                            else:
                                st.error("Agent 3 failed to create the final narrative.")
                    else:
                        st.warning("Agent 2 found no spots that matched the vibe.")
                else:
                    st.warning("Agent 1 found no spots matching your inverse filter. Try increasing the 'Max Review Count'.")
