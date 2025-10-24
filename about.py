from flask import Blueprint, render_template

# Blueprint banaya jisse modular structure rahe
about_bp = Blueprint("about", __name__)

# /about route
@about_bp.route("/about")
def about():
    return render_template("about.html")   # ye templates/about.html load karega
