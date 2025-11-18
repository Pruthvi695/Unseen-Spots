# Unseen-Spots
Uses AI inverse-filtering (high rating, low review count) to find authentic, crowd-free local gems for personalized travel.
Problem Statement: Combating Tourist Saturation with Inverse Filtering

The modern travel landscape is dominated by mass tourism, where centralized platforms and search algorithms consistently surface the same globally famous landmarks. This creates a "paradox of popularity": as more people travel, they are funneled to a limited number of destinations, leading to over-tourism, diminished local authenticity, and degraded user experiences (long queues, high prices, lack of genuine cultural interaction).

Our project aims to solve this by fundamentally shifting the recommendation paradigm. Instead of using traditional Collaborative Filtering (recommending what is most popular), we employ Inverse Data Filtering to surface high-quality experiences that are locally loved but globally unseen. This is an important problem because it:

Enhances Traveler Experience: Provides authentic, crowd-free, and personalized moments.
Supports Local Economies: Distributes tourist revenue to smaller, independent businesses.
Promotes Sustainability: Alleviates pressure on over-stressed historical sites and infrastructure.
Why Agents? The Need for Specialized, Sequential Reasoning

This problem cannot be solved by a single LLM call or a simple database query; it requires a complex, multi-step workflow involving data analysis, qualitative judgment, and creative output. Multi-Agent Systems (MAS) are the ideal solution because they allow for the essential specialization and delegation of these distinct tasks:

Agent Role	Task Complexity	Reason Why an Agent is Best
Tourist Trap Identification	Algorithmic, Data-Intensive	Requires Tool Use (Maps API) and specialized, rigid logic (Inverse Filter) that an LLM alone cannot guarantee.
Vibe Analysis	Qualitative, Judgment-Based	Requires Advanced Reasoning (LLM) to extract subtle atmospheric themes and perform a qualitative match against the user's free-form "vibe" request.
Narrative Generation	Creative, Output-Focused	Requires a Generative AI model to synthesize data and produce a persuasive, engaging, journalist-style narrative, which is best handled by an agent focused purely on creative output.
By using agents, we ensure modularity and robustness. If the data retrieval tool fails, the other agents are not immediately corrupted; they operate on a controlled pipeline, allowing for clear error handling and independent debugging.

What You Created: A Three-Phase Agentic Architecture

The Unseen Spots Discovery Agent is built as a sequential workflow managed by an orchestration framework like LangGraph or CrewAI, leveraging Google's Gemini models for the complex reasoning and generation phases.

The overall architecture is defined by three primary nodes that pass a centralized state object containing the filtered data: .

1. The Tourist Trap Identification Agent (The Inverse Filter)

Role: Data Retrieval and Algorithmic Pruning.
Tool Use: Calls the Google Maps Places API to search around a landmark or city center.
Core Logic: Applies the Inverse Filter: Rating >= 4.5 Stars AND Total Review Count < 500. This aggressively discards all high-volume, highly-saturated tourist spots, regardless of their star rating, to focus only on highly-rated local gems.
2. The Vibe Analysis Agent

Role: Qualitative Matching and Feature Extraction.
Generative AI Task: Uses a Gemini LLM and structured output (Pydantic) to analyze the raw review snippets for the filtered spots.
Output: Converts unstructured text reviews into structured data (e.g., ["cozy", "minimalist"] adjectives and unique features like ["cash-only," "dog-friendly"]). This structure allows the agent to score and rank spots based on how well they match the user's requested "vibe."
3. The Itinerary Builder & Narrative Agent

Role: Synthesis, Optimization, and Presentation.
Logic: Selects the top 3-5 spots based on the Vibe Match Score and checks for logistical grouping (proximity).
Generative AI Task: Uses the Gemini LLM again to transform the structured data into a compelling, personalized narrative for each spot, acting as a travel journalist. It also integrates the final map link for the user.
Demo: Example Output (Lisbon, Portugal)

Input	Vibe Match Score	Final Output
City: Lisbon, Portugal	High Match	1. Livraria A.F. (Rating 4.8 / 120 Reviews)
Vibe: "Hidden, quiet, vintage bookstores with good coffee."		Pitch Narrative: Tucked away on a quiet street near the bustling Bairro Alto, this bookstore is less a shop and more a relic. The air smells of aged paper and espresso. You won't find bestsellers here, but rather a curated collection of local poetry and history, perfect for a slow afternoon of discovery.
Inverse Filter: Review Count \< 500		2. Café do Tempo (Rating 4.6 / 310 Reviews)
Traditional Search (Filtered Out)		Pitch Narrative: Follow the scent of cinnamon and quiet conversation to this small, unmarked cafe. Known only to locals, it's the ideal spot to journal or read your new book, free from the tourist crowds descending on the main squares.
The Build: Technologies and Implementation Details

The project relies heavily on the synergistic use of specific tools:

Orchestration (LangGraph/CrewAI): Used to define the stateful, sequential workflow. This ensures that the output of the Tourist Trap Identification Agent (the filtered list) is reliably passed as input to the Vibe Analysis Agent, preventing workflow failures.
Generative Model (Gemini API): Chosen for its advanced reasoning and reliable structured output capabilities.
Reasoning: Crucial for the Vibe Analysis Agent to interpret sentiment and theme from unstructured review text.
Structured Output: Utilized to enforce the SpotVibe Pydantic schema, guaranteeing the LLM's analysis is returned in a predictable JSON format, which is essential for algorithmic scoring and seamless hand-off between agents.
Data Layer (Google Maps Places API): Provides the necessary geospatial tools, including initial location searching, review count data, star ratings, and raw review text, which serves as the RAG (Retrieval-Augmented Generation) data source for the Vibe Analysis Agent.
Backend/Frontend (Python/Streamlit): Python provides the logical backbone, while Streamlit was used to quickly create a compelling, interactive web demonstration, allowing users to adjust the key variables (Max Review Count, Min Rating) to see the Inverse Filter in action.
If I Had More Time, This Is What I'd Do

Dynamic Threshold Agent: Currently, the Max Review Count threshold (e.g., 500) is static. In a larger project, I would implement a new agent that uses a demographic or geospatial API to dynamically adjust this threshold based on the population density of the target city. A spot with 500 reviews in New York City is "unseen," but 500 reviews in a small village is highly saturated.
Multimodal Vibe Input: Integrate an image analysis tool (e.g., using Gemini's multimodal capabilities) to allow users to upload a reference image (e.g., a photo of a moody cafe interior) instead of just a text prompt for the desired "vibe." The Vibe Analysis Agent would then match the style of the reference image to the analyzed reviews.
Proactive Re-planning Agent: Implement a loop (using LangGraph's conditional edges) that can check for real-time conditions (weather API, temporary closures) on the morning of a planned trip and autonomously re-run the itinerary, generating a contingency plan if a spot is inaccessible.
