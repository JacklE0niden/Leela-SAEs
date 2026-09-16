import React from "react";
import ReactDOM from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "./globals.css";
import { Navigate, RouterProvider, createBrowserRouter } from "react-router-dom";
import { AppStateProvider } from "./contexts/AppStateContext";
import { FeaturesPage } from "@/routes/features/page";
import { DictionaryPage } from "@/routes/dictionaries/page";
import BookmarksPage from "@/routes/bookmarks/page";
import { CircuitsPage } from "@/routes/circuits/page";
import { PlayGamePage } from "@/routes/play-game/page";
import { SemanticSupernodeGraphPage } from "@/routes/semantic-supernode-graph/page";
import { InteractionCircuitPage } from "@/routes/interaction-circuit/page";

const router = createBrowserRouter([
  { path: "/features", element: <FeaturesPage /> },
  { path: "/dictionaries", element: <DictionaryPage /> },
  { path: "/bookmarks", element: <BookmarksPage /> },
  { path: "/circuits", element: <CircuitsPage /> },
  { path: "/play-game", element: <PlayGamePage /> },
  {
    path: "/semantic-supernode-graph",
    element: <SemanticSupernodeGraphPage />,
  },
  { path: "/interaction-circuit", element: <InteractionCircuitPage /> },
  { path: "*", element: <Navigate to="/play-game#circuit-tracing" replace /> },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppStateProvider>
      <RouterProvider router={router} />
    </AppStateProvider>
  </React.StrictMode>,
);
