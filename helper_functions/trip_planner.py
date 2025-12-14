"""Trip planner using OpenAI-compatible API."""
from helper_functions.openai_compat import get_openai_client, get_chat_model


def generate_trip_plan(city, days):
    """
    Generate a detailed trip itinerary for a given city.
    
    Args:
        city: Destination city
        days: Number of days for the trip
    
    Returns:
        str: Detailed trip itinerary or error message
    """
    client = get_openai_client()
    model = get_chat_model("fast")
    
    # Check if the days input is a number and throw an error if it is not
    try:
        days = int(days)

        tripsystem_prompt = [{
            "role": "system",
            "content": "You are a world class trip planner who is knowledgeable about all the tourist attractions in the world. You will serve the user by planning a trip for them and respecting all their preferences."
        }]
        conversation = tripsystem_prompt.copy()
        user_message = f"Craft a thorough and detailed travel itinerary for {city}. This itinerary should encompass the city's most frequented tourist attractions, as well as its top-rated restaurants, all of which should be visitable within a timeframe of {days} days including the best budget-friendly hotels/resorts to stay for {days-1} nights. The itinerary should be strategically organized to take into account the distance between each location and the time required to travel there, maximizing efficiency. Moreover, please include specific time windows for each location, arranged in ascending order, to facilitate effective planning. The final output should be a numbered list, where each item corresponds to a specific location. Accompany each location with a brief yet informative description to provide context and insight."
        conversation.append({"role": "user", "content": str(user_message)})
        
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            max_tokens=2048,
            temperature=0.3,
        )
        message = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": str(message)})
        return f"Here is your trip plan for {city} for {days} day(s): {message}"
    except Exception:
        return "Please enter a number for days."
