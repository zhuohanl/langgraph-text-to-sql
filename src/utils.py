from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from PIL import Image as PILImage
import io

##########################
### Utils
##########################

# Utils
def generate_mermaid(app, output_image_name):
    # Assuming `app.get_graph()` is your graph object
    graph = app.get_graph()

    # Generate the Mermaid diagram as a PNG image using MermaidDrawMethod.API
    png_data = graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)

    # Save the PNG image locally
    output_file_path = output_image_name  # Specify the desired file path

    # Convert the binary PNG data into an image using PIL
    image = PILImage.open(io.BytesIO(png_data))
    image.save(output_file_path)

    print(f"Mermaid diagram saved to {output_file_path}")