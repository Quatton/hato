observation_start = "<observations>"
observation_end = "</observations>"
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
answer_start = "<answer>"
answer_end = "</answer>"
ward_start = "<ward>"
ward_end = "</ward>"
town_start = "<town>"
town_end = "</town>"

system_prompt = f"""You are given an image of a location in Tokyo.
Make observations based on:
- Building types (residential/commercial), architectural era, density, height restrictions
- Vegetation type and abundance (native vs planted species)
- Road infrastructure: width, markings, materials, surface condition
- Municipal features: lamp styles, signage, utility poles (try to identify lamp colors and geometry)
- Landmarks or distinctive features, especially those unique to Tokyo
- Urban planning patterns: street layout, block size
- Topography: elevation, proximity to hills/water
- Geographical context: proximity to major roads, rivers, parks
- Never include any town names here to reduce biases
Place it between {observation_start} and {observation_end}

Then, based on your observation make a reasoning to come up with candidates for wards
and towns in Tokyo that likely match the description. You can reason in English for now. Place it between {reasoning_start} and {reasoning_end}.

Finally, provide your final answer as XML format between {answer_start}{ward_start}(required){ward_end}{town_start}(optional){town_end}{answer_end}
"""
