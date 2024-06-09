import random
import json

# Fixed seed for reproducibility
random.seed(42)

# List of possible purchases
possible_purchases = ['amazon', 'flipkart', 'myntra']

def generate_purchase_history_balanced(num_users, max_purchases_per_user):
    
    data = []
    num_categories = len(possible_purchases)
    purchases_per_category = (num_users * max_purchases_per_user) // num_categories

    category_count = {category: 0 for category in possible_purchases}
    for i in range(1, num_users + 1):
        user_id = f'user{i}'
        num_purchases = random.randint(1, min(max_purchases_per_user, num_categories))
        purchases = random.sample(possible_purchases, num_purchases)
        
        for purchase in purchases:
            category_count[purchase] += 1
            
        data.append({'user': user_id, 'purchases': purchases})
        
    return data

def save_data_to_file(data, filename):
    
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    # Parameters
    num_users = 1000
    max_purchases_per_user = 10
    
    # Generate the data
    simulated_data = generate_purchase_history_balanced(num_users, max_purchases_per_user)
    
    # Save the data to a file
    save_data_to_file(simulated_data, 'simulated_purchase_history.json')

    print(f"Generated data for {num_users} users and saved to 'simulated_purchase_history.json'.")
