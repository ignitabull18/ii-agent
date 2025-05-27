if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "Using Vertex Client"
  echo "PROJECT_ID: $PROJECT_ID"
  echo "REGION: $REGION"
  exec xvfb-run --auto-servernum python ws_server.py --use-container-workspace --project-id $PROJECT_ID --region $REGION
else
  echo "Using Anthropic Client, reading ANTHROPIC_API_KEY from .env"
  exec xvfb-run --auto-servernum python ws_server.py --use-container-workspace
fi