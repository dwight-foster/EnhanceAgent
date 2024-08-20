def call_tool(client, tool_calls, response_message, messages, tool):
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {

            "zoom_in": tool.zoom,
            "describe": tool.describe,
            "upscale_image": tool.upscale_image,
            "zoom_out": tool.zoom_out
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            function_to_call = available_functions[function_name]
            function_args = tool_call['function']['arguments']

            if function_name == "zoom_in":
                img, function_response = function_to_call(
                    text=function_args["text"])
            elif function_name == "describe":
                img, function_response = function_to_call()
            elif function_name == "upscale_image":
                img, function_response = function_to_call()
            elif function_name == "zoom_out":
                img, function_response = function_to_call()

            messages.append(
                {
                    'role': 'tool',
                    'content': function_response,
                }
            )

        # Second API call: Get final response from the model
        final_response = client.chat(model="dwightfoster03/functionary-small-v3.1",
                                     messages=messages)  # get a new response from the model where it can see the function response
        return img, final_response['message']