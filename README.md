# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/REPLACE_WITH_PROJECT_ID

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/REPLACE_WITH_PROJECT_ID) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/REPLACE_WITH_PROJECT_ID) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)

## Run With Vercel Backend

This project now includes a Vercel Serverless backend route at `api/hf-inference.ts` used by all playgrounds.

### Local development

1. Install deps:

```sh
npm i
```

2. Install Vercel CLI (one-time):

```sh
npm i -g vercel
```

3. Run frontend + API together:

```sh
vercel dev
```

### Required environment variables

Set in Vercel Project Settings -> Environment Variables:

- `HUGGINGFACE_API_KEY` (required for HF Inference API mode)
- `INFERENCE_BACKEND_URL` (recommended; if set, API proxies to your RunPod backend)

### Deployment (frontend + backend)

1. Push this repo to GitHub.
2. Import it in Vercel.
3. Framework preset: `Vite`.
4. Build command: `npm run build`
5. Output directory: `dist`
6. Deploy.

## Recommended Production Architecture

For a student-facing CV lab with heavier models/video:

- Frontend on Vercel (React app)
- API proxy on Vercel (`/api/hf-inference`)
- GPU backend on RunPod (`backend/` folder)

Flow:

1. Browser uploads image/video to Vercel API.
2. Vercel API forwards to RunPod `/infer`.
3. RunPod executes Hugging Face models on GPU and returns predictions.
4. Frontend renders outputs + educational explanations/checklists.

RunPod backend setup instructions are in:

- `backend/README.md`
- includes both Docker and non-Docker (direct Python) options

### GPU baseline for RunPod

- CUDA: **12.1**
- Minimum: **16 GB VRAM**
- Recommended for chained perception playground: **24 GB VRAM**
