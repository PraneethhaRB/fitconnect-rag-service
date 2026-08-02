from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from groq import Groq
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = FastAPI(title="FitConnect RAG Service")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ─── Embeddings via HuggingFace Inference API (no local torch/onnxruntime) ────
# This avoids the SIGILL crashes caused by torch/onnxruntime assuming CPU
# instructions (AVX2/AVX-512) that Render's instance doesn't support.
# Get a free token at https://huggingface.co/settings/tokens and set it as
# the HF_API_TOKEN environment variable on Render.

HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)
HF_TOKEN = os.environ.get("HF_API_TOKEN")


class HFEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": input, "options": {"wait_for_model": True}},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


embedding_function = HFEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="fitness_knowledge",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)

FITNESS_KNOWLEDGE = [
    {
        "id": "protein_muscle",
        "text": """Protein requirements for muscle gain (hypertrophy):
        Consume 1.6 to 2.2 grams of protein per kilogram of bodyweight daily.
        This is the evidence-based range for maximizing muscle protein synthesis.
        Caloric surplus of 200 to 300 calories supports muscle growth without
        excess fat gain. Distribute protein across 4 to 6 meals throughout the day.
        Complete proteins including meat, eggs, dairy, and soy provide all essential
        amino acids needed for muscle synthesis and recovery.""",
        "category": "nutrition",
        "goal": "BUILD_MUSCLE"
    },
    {
        "id": "calories_weight_loss",
        "text": """Calorie deficit for weight loss:
        A caloric deficit of 300 to 500 calories per day creates sustainable
        fat loss of approximately 0.3 to 0.5 kilograms per week. Protein intake
        should be 1.6 to 2.2 grams per kg of bodyweight to preserve muscle mass
        during weight loss. Avoid deficits larger than 1000 calories as they
        cause muscle loss and metabolic adaptation. Track calories using a food
        diary or app for the first few weeks to calibrate your intake.""",
        "category": "nutrition",
        "goal": "LOSE_WEIGHT"
    },
    {
        "id": "workout_frequency",
        "text": """Workout frequency and training volume:
        Research supports training each muscle group 2 times per week for optimal
        hypertrophy. A push pull legs split or upper lower split achieves this
        efficiently. Beginners benefit from 3 full body sessions per week.
        Advanced lifters may train 4 to 6 days. Rest periods of 48 to 72 hours
        between training the same muscle group prevents overtraining and allows
        adequate recovery for strength and size gains.""",
        "category": "training",
        "goal": "BUILD_MUSCLE"
    },
    {
        "id": "cardio_endurance",
        "text": """Cardiovascular training and endurance:
        The American Heart Association recommends 150 minutes of moderate intensity
        or 75 minutes of vigorous cardio per week for general health. Zone 2 training
        at 60 to 70 percent of maximum heart rate builds aerobic base efficiently
        and improves fat oxidation. High intensity interval training HIIT improves
        VO2 max and burns more calories in less time. For running specifically,
        increase weekly mileage by no more than 10 percent per week to prevent injury.""",
        "category": "training",
        "goal": "ENDURANCE"
    },
    {
        "id": "sleep_recovery",
        "text": """Sleep and recovery for athletic performance:
        7 to 9 hours of sleep per night is optimal for athletic performance and
        muscle recovery. During deep sleep growth hormone is released which repairs
        muscle tissue damaged during training. Sleep deprivation reduces testosterone
        levels, increases cortisol which breaks down muscle, impairs protein synthesis,
        and decreases reaction time and decision making ability. Prioritize sleep
        consistency — same bedtime and wake time daily — over total duration.""",
        "category": "recovery",
        "goal": "GENERAL_FITNESS"
    },
    {
        "id": "nutrition_general",
        "text": """Balanced nutrition for fitness:
        A fitness-supporting diet includes complex carbohydrates such as oats, rice,
        and sweet potato for sustained energy, lean protein including chicken, fish,
        and eggs for muscle repair, healthy fats from avocado, nuts, and olive oil
        for hormone production, and abundant vegetables for micronutrients and fiber.
        Timing matters: consume carbohydrates around workouts for energy and recovery,
        distribute protein evenly throughout the day, and reduce simple carbohydrates
        in the evening when energy demands are lower.""",
        "category": "nutrition",
        "goal": "HEALTHY_LIFESTYLE"
    },
    {
        "id": "weight_loss_training",
        "text": """Training for weight loss:
        Combining resistance training with cardio is most effective for fat loss.
        Resistance training preserves muscle mass during a caloric deficit and
        raises resting metabolic rate. High intensity interval training burns more
        calories in less time than steady state cardio. Aim for 3 to 4 training
        sessions per week combining both modalities. Non exercise activity
        thermogenesis NEAT — steps, walking, standing — accounts for significant
        daily calorie burn and should not be neglected.""",
        "category": "training",
        "goal": "LOSE_WEIGHT"
    },
    {
        "id": "hydration",
        "text": """Hydration for exercise and performance:
        Drink 35 to 45 milliliters of water per kilogram of bodyweight daily.
        During exercise consume 150 to 250 milliliters every 15 to 20 minutes.
        Even 2 percent dehydration reduces athletic performance measurably.
        Signs of adequate hydration include pale yellow urine. Electrolytes
        sodium potassium and magnesium are lost through sweat and should be
        replaced during sessions longer than 60 minutes through sports drinks
        or electrolyte supplements.""",
        "category": "nutrition",
        "goal": "GENERAL_FITNESS"
    },
    {
        "id": "progressive_overload",
        "text": """Progressive overload for strength and muscle gains:
        Progressive overload means consistently increasing the demands placed on
        your muscles over time. This can mean adding weight, doing more repetitions,
        reducing rest periods, or improving technique. Without progressive overload
        the body adapts and gains stall. Track your workouts to ensure progression.
        Increase load by 2.5 to 5 kilograms when you can complete all sets with
        good form. Linear progression works best for beginners. Intermediate lifters
        benefit from weekly or monthly progression cycles.""",
        "category": "training",
        "goal": "BUILD_MUSCLE"
    },
    {
        "id": "flexibility_mobility",
        "text": """Flexibility and mobility for healthy lifestyle:
        Regular stretching and mobility work reduces injury risk and improves
        movement quality. Dynamic stretching before training prepares muscles
        and joints. Static stretching after training when muscles are warm
        improves flexibility over time. Hold static stretches for 30 to 60 seconds.
        Yoga and Pilates are excellent for combining flexibility strength and
        mindfulness. Focus on hip flexors thoracic spine and shoulder mobility
        which are commonly restricted in people with sedentary jobs.""",
        "category": "recovery",
        "goal": "HEALTHY_LIFESTYLE"
    }
]


def seed_knowledge_base():
    existing = collection.count()
    if existing >= len(FITNESS_KNOWLEDGE):
        print(f"Knowledge base already seeded with {existing} chunks")
        return

    print(f"Seeding knowledge base with {len(FITNESS_KNOWLEDGE)} chunks...")

    collection.add(
        ids=[chunk["id"] for chunk in FITNESS_KNOWLEDGE],
        documents=[chunk["text"] for chunk in FITNESS_KNOWLEDGE],
        metadatas=[
            {"category": chunk["category"], "goal": chunk["goal"]}
            for chunk in FITNESS_KNOWLEDGE
        ]
    )
    print("Knowledge base seeded successfully")


seed_knowledge_base()


class QuestionRequest(BaseModel):
    question: str
    user_goal: str = "general fitness"
    goal_category: str = "GENERAL_FITNESS"
    top_k: int = 3


class AnswerResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]
    retrieval_method: str = "vector_similarity"


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    # Step 1: RETRIEVE — pure vector similarity
    results = collection.query(
        query_texts=[request.question],
        n_results=request.top_k
    )

    retrieved_texts = results["documents"][0]
    retrieved_ids = results["ids"][0]
    distances = results["distances"][0]

    print(f"\nQuery: {request.question}")
    for i, (chunk_id, dist) in enumerate(zip(retrieved_ids, distances)):
        print(f"  {i+1}. {chunk_id} (similarity: {1-dist:.3f})")

    if not retrieved_texts:
        return AnswerResponse(
            answer="I don't have specific information on that topic. "
                   "For personalized advice, consult a certified fitness professional.",
            retrieved_chunks=[],
            retrieval_method="fallback"
        )

    # Step 2: AUGMENT
    context = "\n\n---\n\n".join(retrieved_texts)

    prompt = f"""You are a knowledgeable fitness advisor for FitConnect.

Answer the user's question using ONLY the information provided in the context below.
If the context doesn't fully cover the question, say so clearly.
Be specific and practical. Answer in 2 to 4 sentences.

User's current fitness goal: {request.user_goal}

Retrieved fitness knowledge:
---
{context}
---

User's question: {request.question}

Answer:"""

    # Step 3: GENERATE
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a fitness advisor. Answer questions "
                           "based ONLY on the provided context. Be concise and practical."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="openai/gpt-oss-120b",
        temperature=0.3,
        max_tokens=250
    )

    answer = chat_completion.choices[0].message.content

    return AnswerResponse(
        answer=answer,
        retrieved_chunks=retrieved_ids,
        retrieval_method="vector_similarity"
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "chunks_indexed": collection.count(),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2 (via HF Inference API)",
        "vector_db": "Chroma",
        "llm": "Groq openai/gpt-oss-120b"
    }


@app.post("/debug/retrieve")
async def debug_retrieve(request: QuestionRequest):
    results = collection.query(
        query_texts=[request.question],
        n_results=request.top_k
    )
    return {
        "question": request.question,
        "retrieved": [
            {
                "id": results["ids"][0][i],
                "similarity": round(1 - results["distances"][0][i], 4),
                "text_preview": results["documents"][0][i][:150] + "...",
                "metadata": results["metadatas"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
