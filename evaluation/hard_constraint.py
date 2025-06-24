

from tools.attractions.apis import Attractions
from tools.restaurants.apis import Restaurants
from tools.hotels.apis import Accommodations
import sys
sys.path.append("../")

try:
    from utils import load_json_file
except:
    from evaluation.utils import load_json_file


accommodation = Accommodations()
restaurants = Restaurants()
attractions = Attractions()


def calc_cost_from_itinerary_wo_intercity(itinerary, people_number):
    total_cost = 0
    for day in itinerary:
        for activity in day["activities"]:

            for transport in activity.get("transports", []):

                mode = transport["mode"]
                if mode == 'taxi':
                    if 'cars' in transport.keys():
                        total_cost += transport.get('cars', 0) * \
                            transport.get("cost", 0)
                    else:
                        total_cost += transport.get('tickets', 0) * \
                            transport.get("cost", 0)
                if mode == 'metro':
                    total_cost += transport.get('tickets', 0) * \
                        transport.get("cost", 0)

            # if activity["type"] == "airplane":
            #     total_cost += activity.get('tickets',0)*activity.get("cost", 0)

            # if activity["type"] == "train":
            #     total_cost += activity.get('tickets',0)*activity.get("cost", 0)

            if activity["type"] == "breakfest" or activity["type"] == "lunch" or activity["type"] == "dinner":
                total_cost += activity.get('cost', 0)*people_number

            # if activity["type"] == "accommodation":
            #     total_cost += activity.get('rooms',0)*activity.get("cost", 0)

            if activity["type"] == "attraction":
                total_cost += activity.get('tickets', 0) * \
                    activity.get("cost", 0)
    return total_cost


def get_symbolic_concepts(symbolic_input, plan_json):

    target_city = symbolic_input['target_city']
    start_city = symbolic_input['start_city']

    # Extracting basic information
    days = len(plan_json["itinerary"])
    people_number = plan_json["people_number"]

    # Calculating total cost
    total_cost = 0
    transport_types = set()
    intercity_transport = set()
    train_type = set()

    food_type = set()
    spot_type = set()
    hotel_feature = set()
    hotel_names = set()
    restaurant_names = set()
    attraction_names = set()

    # rooms and room_type are INT
    rooms = 0
    tickets = 0
    room_type = 0
    food_prices = []
    hotel_prices = []

    for day in plan_json["itinerary"]:
        for activity in day["activities"]:
            if 'tickets' in activity:
                tickets = activity.get('tickets', 0)
            for transport in activity.get("transports", []):
                if 'tickets' in transport.keys():
                    tickets = activity.get('tickets', 0)

                mode = transport["mode"]
                if mode == 'taxi':
                    if 'cars' in transport.keys():
                        total_cost += transport.get('cars', 0) * \
                            transport.get("cost", 0)
                    else:
                        total_cost += transport.get('tickets', 0) * \
                            transport.get("cost", 0)
                if mode == 'metro':
                    total_cost += transport.get('tickets', 0) * \
                        transport.get("cost", 0)

                if mode in ['metro', 'taxi']:
                    transport_types.add(mode)
                if mode == "walk" and len(activity.get("transports", [])) == 1 and transport.get('distance', 2) > 1:
                    transport_types.add(mode)

            if activity["type"] == "airplane":
                total_cost += activity.get('tickets', 0) * \
                    activity.get("cost", 0)
                intercity_transport.add("airplane")

            if activity["type"] == "train":
                total_cost += activity.get('tickets', 0) * \
                    activity.get("cost", 0)
                intercity_transport.add("train")
                train_id = activity.get("TrainID", "")
                if train_id:
                    train_type.add(train_id[0])

            if activity["type"] == "breakfest" or activity["type"] == "lunch" or activity["type"] == "dinner":
                select_food_type = restaurants.select(
                    target_city, key='name', func=lambda x: x == activity["position"])['cuisine']
                if not select_food_type.empty:
                    food_type.add(select_food_type.iloc[0])
                restaurant_names.add(activity["position"])
                food_prices.append(activity["cost"])
                total_cost += activity.get('cost', 0)*people_number

            if activity["type"] == "accommodation":
                select_hotel_type = accommodation.select(
                    target_city, key='name', func=lambda x: x == activity["position"])['featurehoteltype']
                if not select_hotel_type.empty:
                    hotel_feature.add(select_hotel_type.iloc[0])
                hotel_names.add(activity["position"])
                hotel_prices.append(activity["cost"])
                total_cost += activity.get('rooms', 0)*activity.get("cost", 0)
                rooms = activity.get('rooms', 0)
                room_type = activity.get('room_type', 1)

            if activity["type"] == "attraction":
                select_attraction_type = attractions.select(
                    target_city, key='name', func=lambda x: x == activity["position"])['type']
                if not select_attraction_type.empty:
                    spot_type.add(select_attraction_type.iloc[0])
                attraction_names.add(activity["position"])
                total_cost += activity.get('tickets', 0) * \
                    activity.get("cost", 0)

    # Calculating average food and hotel prices
    food_price = sum(food_prices) / len(food_prices) if food_prices else 0
    hotel_price = sum(hotel_prices) / len(hotel_prices) if hotel_prices else 0

    # change tickets to int

    return {
        "days": days,
        "people_number": people_number,
        "cost": total_cost,
        "transport_type": transport_types,
        "intercity_transport": intercity_transport,
        "train_type": train_type,
        "food_type": food_type,
        "spot_type": spot_type,
        "hotel_feature": hotel_feature,
        "food_price": food_price,
        "hotel_price": hotel_price,
        "hotel_names": hotel_names,
        "restaurant_names": restaurant_names,
        "attraction_names": attraction_names,
        "tickets": tickets,
        "rooms": rooms,
        "room_type": room_type
    }


def evaluate_constraints(extracted_vars, hard_logic):
    if extracted_vars is None:
        return [False]*len(hard_logic)

    results = []
    for constraint in hard_logic:
        try:
            # Evaluate the constraint in a safe manner
            result = eval(constraint, {"__builtins__": None}, extracted_vars)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating constraint '{constraint}': {e}")
            results.append(False)
    return results


def calculate_metrics(results_list):
    total_constraints = sum(len(results) for results in results_list)
    satisfied_constraints = sum(sum(results) for results in results_list)

    macro_accuracy = sum(all(results)
                         for results in results_list) / len(results_list)
    micro_accuracy = satisfied_constraints / total_constraints

    return macro_accuracy*100, micro_accuracy*100, results_list


def evaluate_hard_constraints(symbolic_input_list, plan_json_list):
    assert len(symbolic_input_list) == len(plan_json_list)
    results = []
    for (symbolic_input, plan_json) in zip(symbolic_input_list, plan_json_list):
        try:
            extracted_vars = get_symbolic_concepts(symbolic_input, plan_json)
        except:
            extracted_vars = None
        results.append(evaluate_constraints(
            extracted_vars, symbolic_input["hard_logic"]))
    return calculate_metrics(results)


if __name__ == "__main__":

    symbolic_input_list = []
    plan_json_list = []

    for i in range(5):
        test_plan_path = './example/plan_{}.json'.format(i+1)
        test_example_path = './example/query_{}.json'.format(i+1)
        test_example = load_json_file(test_example_path)
        test_plan = load_json_file(test_plan_path)
        symbolic_input_list.append(test_example)
        plan_json_list.append(test_plan)
    macro_accuracy, micro_accuracy, _ = evaluate_hard_constraints(
        symbolic_input_list, plan_json_list)
    print('macro: {}%, micro: {}%'.format(macro_accuracy, micro_accuracy))
