import json

def call_tool(client, tool_calls, response_message, messages, tool):
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {

            "zoom": tool.zoom,
            "describe": tool.describe,
            "upscale_image": tool.upscale_image,
            "zoom_out": tool.zoom_out
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "zoom":
                img, function_response = function_to_call(
                    text=function_args.get("text"))
            elif function_name == "describe":
                img, function_response = function_to_call()
            elif function_name == "upscale_image":
                img, function_response = function_to_call()
            elif function_name == "zoom_out":
                img, function_response = function_to_call()

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return img, second_response.choices[0].message