import django
import copy
import logging
from django.conf import settings
from django.core.exceptions import AppRegistryNotReady
from human_feedback_site import settings as site_settings

def initialize():
    try:
        new = copy.copy(site_settings.__dict__)
        to_remove = [key for key in new if not key.isupper()]
        for key in to_remove:
            del new[key]

        settings.configure(**new)
        django.setup()
    except RuntimeError:
        logging.warning("Tried to double configure the API, ignore this if running the Django app directly")

initialize()

try:
    from human_feedback_api.models import Comparison
except AppRegistryNotReady:
    logging.info("Could not yet import Feedback")
