import os
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_pymongo import PyMongo
from langchain.llms import openai
from langchain.llms import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import GoogleSerperAPIWrapper
from datetime import datetime
from bson import ObjectId
# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure MongoDB
app.config["MONGO_URI"] = os.environ.get("MONGODB_URI")
mongo = PyMongo(app)

# Configure OpenAI (SambaNova) API
openai.api_key = os.environ.get("SAMBANOVA_API_KEY")
openai.api_base = "https://api.sambanova.ai/v1"
Serper_API_KEY = os.environ.get("SERPER_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    model_name='Llama-3.2-90B-Vision-Instruct',
    temperature=0.1,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base
)
# Agent Classes
class CommunityEducationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=['community_type', 'disaster_type'],
            template="""As a disaster preparedness expert, develop a comprehensive guide tailored for a {community_type} community, addressing the risks associated with {disaster_type}. The guide should include:

1. An overview of the specific risks and challenges posed by {disaster_type} in {community_type} communities.
2. Detailed preparedness steps that individuals and community organizations can take.
3. Cultural, social, and economic factors to consider for effective engagement.
4. Resources and recommendations for enhancing community resilience.

Ensure the guide is accessible, actionable, and considers the unique characteristics of {community_type} communities."""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def generate_content(self, community_type, disaster_type):
        return self.chain.run({
            'community_type': community_type, 
            'disaster_type': disaster_type
        })

class RiskAssessmentAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=['location', 'disaster_type'],
            template="""You are a risk assessment specialist. Provide a detailed risk assessment report for {location} focusing on {disaster_type}. The report should cover:

1. Current and historical data on {disaster_type} occurrences in {location}.
2. Geographical and environmental factors contributing to vulnerability.
3. Potential impact scenarios and affected areas.
4. Predictive models or forecasts indicating future risks.
5. Recommended strategies for mitigation and preparedness.

Present the information in a clear and professional manner suitable for local authorities and community leaders."""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def assess_risks(self, location, disaster_type):
        return self.chain.run({
            'location': location, 
            'disaster_type': disaster_type
        })

class EmergencyResponseAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=['scenario', 'community_size'],
            template="""As an emergency response coordinator, develop a comprehensive response plan for a {community_size} community facing a {scenario} scenario. The plan should include:

1. Immediate actions to take upon awareness of the {scenario}.
2. Evacuation routes and shelter locations.
3. Communication plans for disseminating information to the public.
4. Coordination with local agencies and emergency services.
5. Special considerations for vulnerable populations.
6. Post-event recovery and support strategies.

Ensure the plan is practical, detailed, and considers the specific needs of a {community_size} community."""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def create_response_plan(self, scenario, community_size):
        return self.chain.run({
            'scenario': scenario, 
            'community_size': community_size
        })

class LiveUpdatesAgent:
    def __init__(self, llm):
        self.llm = llm
        self.search = GoogleSerperAPIWrapper(serper_api_key=Serper_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=['location', 'search_results'],
            template="""You are a disaster alert assistant. Based on the following search results for {location}, summarize the live updates and alerts related to any disasters or emergencies. Be concise and provide actionable information.

Search Results:
{search_results}
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def get_live_updates(self, location):
        # Use the search wrapper to get live updates
        query = f"{location} disaster alerts"
        search_results = self.search.run(query)
        # Run the LLM chain with the search results
        return self.chain.run({
            'location': location,
            'search_results': search_results
        })

class DisasterHistoryAgent:
    def __init__(self, llm):
        self.llm = llm
        self.search = GoogleSerperAPIWrapper(serper_api_key=Serper_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=['location', 'search_results'],
            template="""You are a historian specializing in natural disasters. Based on the following search results for {location}, provide a detailed history of significant natural disasters that have occurred in the region. Include dates, impacts, and any notable aftermaths.

Search Results:
{search_results}
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def get_disaster_history(self, location):
        query = f"History of natural disasters in {location}"
        search_results = self.search.run(query)
        return self.chain.run({
            'location': location,
            'search_results': search_results
        })

class AidResourcesAgent:
    def __init__(self, llm):
        self.llm = llm
        self.search = GoogleSerperAPIWrapper(serper_api_key=Serper_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=['location', 'search_results'],
            template="""You are an advisor on disaster relief resources. Based on the following search results for {location}, provide detailed information about insurance options, government aid programs, and other assistance schemes available to residents in case of a disaster. Include eligibility criteria and how to access these resources.

Search Results:
{search_results}
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def get_aid_resources(self, location):
        query = f"Disaster insurance and government aid schemes in {location}"
        search_results = self.search.run(query)
        return self.chain.run({
            'location': location,
            'search_results': search_results
        })

class NaturalDisasterExpertAgent:
    def __init__(self, llm):
        self.llm = llm
        self.search = GoogleSerperAPIWrapper(serper_api_key=Serper_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=['user_query', 'search_results'],
            template="""You are a natural disaster expert tutor. The user has asked: "{user_query}". Based on the following search results and your expertise, provide a detailed, informative answer suitable for someone new to the topic. Aim to educate the user thoroughly on the subject.

Search Results:
{search_results}

Answer:
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def answer_query(self, user_query):
        # Use the search wrapper to get relevant information
        search_results = self.search.run(user_query)
        # Run the LLM chain with the search results
        return self.chain.run({
            'user_query': user_query,
            'search_results': search_results
        })

# Routes
@app.route("/")
def home():
    # Fetch chat history from MongoDB
    chats = mongo.db.chats.find({})
    myChats = [chat for chat in chats]
    return render_template("index.html", myChats=myChats)

@app.route("/api/generate", methods=["POST"])
def generate_content():
    try:
        data = request.json
        module = data.get('module')
        inputs = data.get('inputs', {})
        result = None

        # Process based on module type
        if module == 'Community Education':
            agent = CommunityEducationAgent(llm)
            result = agent.generate_content(
                inputs.get('community_type'),
                inputs.get('disaster_type')
            )
        # [Add other module conditions here]

        if result:
            # Store in MongoDB
            chat_entry = {
                'module': module,
                'inputs': inputs,
                'response': result,
                'timestamp': datetime.datetime.now()
            }
            mongo.db.chats.insert_one(chat_entry)

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/api/chat-history", methods=["GET"])
def get_chat_history():
    chats = mongo.db.chats.find({}).sort('timestamp', -1)
    chat_list = [{**chat, '_id': str(chat['_id'])} for chat in chats]
    return jsonify({
        'success': True,
        'chat_history': chat_list
    })

@app.route("/api/chat-history", methods=["DELETE"])
def clear_chat_history():
    mongo.db.chats.delete_many({})
    return jsonify({
        'success': True,
        'message': 'Chat history cleared'
    })

@app.route("/api/chat-history/<chat_id>", methods=["DELETE"])
def delete_chat_entry(chat_id):
    try:
        result = mongo.db.chats.delete_one({'_id': ObjectId(chat_id)})
        if result.deleted_count > 0:
            return jsonify({
                'success': True,
                'message': f'Entry {chat_id} deleted'
            })
        return jsonify({
            'success': False,
            'error': 'Entry not found'
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route("/api/live-updates/<location>", methods=["GET"])
def get_live_updates(location):
    try:
        agent = LiveUpdatesAgent(llm)
        result = agent.get_live_updates(location)
        return jsonify({
            'success': True,
            'location': location,
            'updates': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/api/disaster-history/<location>", methods=["GET"])
def get_disaster_history(location):
    try:
        agent = DisasterHistoryAgent(llm)
        result = agent.get_disaster_history(location)
        return jsonify({
            'success': True,
            'location': location,
            'history': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/api/aid-resources/<location>", methods=["GET"])
def get_aid_resources(location):
    try:
        agent = AidResourcesAgent(llm)
        result = agent.get_aid_resources(location)
        return jsonify({
            'success': True,
            'location': location,
            'resources': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/api/risk-assessment", methods=["POST"])
def assess_risks():
    try:
        data = request.json
        location = data.get('location')
        disaster_type = data.get('disaster_type')
        
        if not location or not disaster_type:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400

        agent = RiskAssessmentAgent(llm)
        result = agent.assess_risks(location, disaster_type)
        return jsonify({
            'success': True,
            'assessment': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/api/emergency-response", methods=["POST"])
def create_response_plan():
    try:
        data = request.json
        scenario = data.get('scenario')
        community_size = data.get('community_size')
        
        if not scenario or not community_size:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400

        agent = EmergencyResponseAgent(llm)
        result = agent.create_response_plan(scenario, community_size)
        return jsonify({
            'success': True,
            'response_plan': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/api/expert-query", methods=["POST"])
def expert_query():
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Missing query parameter'
            }), 400

        agent = NaturalDisasterExpertAgent(llm)
        result = agent.answer_query(query)
        return jsonify({
            'success': True,
            'answer': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)