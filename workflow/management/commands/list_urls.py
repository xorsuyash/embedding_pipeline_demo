from django.core.management.base import BaseCommand
from django.urls import get_resolver, URLPattern, URLResolver

class Command(BaseCommand):
    help = 'Displays all the URL mappings of the project.'

    def handle(self, *args, **kwargs):
        resolver = get_resolver()
        url_patterns = resolver.url_patterns

        def list_urls(lis, parent_pattern=None):
            for pattern in lis:
                if isinstance(pattern, URLPattern):
                    full_pattern = f"{parent_pattern}/{pattern.pattern}".replace('//', '/')
                    self.stdout.write(full_pattern)
                elif isinstance(pattern, URLResolver):
                    nested_pattern = f"{parent_pattern}/{pattern.pattern}".replace('//', '/')
                    list_urls(pattern.url_patterns, nested_pattern)

        list_urls(url_patterns, '')


