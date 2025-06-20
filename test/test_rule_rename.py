import os

import ollama

prompt = """
Follow the pattern bellow for suggesting a new descriptive, semantic name for the function, and only give one answer,
 do not write anything after `function is:`, and do not give any explanation.

function is:
def conditions_for_general_pick_up_detector_start_condition_checker(cls_, event, output_):
    return isinstance(event, LossOfContactEvent)
    
Answer is:
def check_for_loss_of_contact(cls_, event, output_):

function is:
def conditions_for_general_pick_up_detector_start_condition_checker(cls_: Type[GeneralPickUpDetector], event: Event, output_: bool) -> bool:
    all_objects = event.tracked_objects
    if isinstance(event, AbstractContactEvent):
        all_objects.extend(event.objects)
    transportable_objects = select_transportable_objects(all_objects)
    if len(transportable_objects) == 0:
        return False
    obj_in_currently_tracked_objects = all(obj in cls_.currently_tracked_objects
                                           and abs(cls_.currently_tracked_objects[obj].starter_event.timestamp
                                                   - event.timestamp) < timedelta(milliseconds=500).total_seconds()
                                           for obj in transportable_objects)
    return obj_in_currently_tracked_objects
Answer is:
def check_if_all_transportable_objects_are_being_tracked(cls_, event: Event, output_: bool) -> bool:

function is:
def conditions_for_general_pick_up_detector_start_condition_checker(cls_: Type[GeneralPickUpDetector],
 event: Event, output_: bool) -> bool:
    contact_points = event.tracked_object.contact_points
    return (any(issubclass(obj.obj_type, Location) for obj in contact_points.get_objects_that_have_points())
            or is_object_supported_by_container_body(event.tracked_object))
Answer is:
"""
"""
Similar to the example above, rename the function bellow with a short, descriptive, semantic name that describes what is
 inside the function, just the name is enough, but make sure the name follows python standards.
"""
# from openai import OpenAI
#
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#
# models = client.models.list()
# print([m.id for m in models.data])
#
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": prompt}]
# )
# print(response.choices[0].message.content)
#
# print(response['choices'][0]['message']['content'])


response = ollama.chat(model="codellama",  messages=[
    {"role": "user", "content": prompt}
])
print(response["message"]["content"].strip().split("(")[0].replace("def ", ""))

# import requests
#
# url = "http://localhost:11434/api/chat"
# headers = {"Content-Type": "application/json"}
#
# payload = {
#     "model": "codellama",
#     "messages": [
#         {"role": "user", "content": prompt}
#     ],
#     "stop": ["function is:"],
#     "stream": False         # required to avoid JSONDecodeError
# }
#
# response = requests.post(url, json=payload, headers=headers)
# print(response.json()["message"]["content"].strip().split("(")[0].replace("def ", ""))

# print(response['message']['content'].strip())