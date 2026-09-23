"""From a named destination to a route, and from a route to a goal pixel.

    building_router.py       multi-floor routing: which floor, which lift,
                             which checkpoints -- and refusals (the stairs)
    checkpoint_projector.py  a map-frame checkpoint -> the pixel in the dog's
                             camera frame that VAMOS is asked to drive at
    visualize_route.py       draw a planned route, for looking at

This is the "map decides WHERE" half of the spec's division of labour. The VLM
decides HOW to get to the pixel this layer hands it.
"""
