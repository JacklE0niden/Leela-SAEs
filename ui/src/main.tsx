import React from "react";
import ReactDOM from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "./globals.css";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import { FeaturesPage } from "@/routes/features/page";
import { RootPage } from "./routes/page";
import { DictionaryPage } from "./routes/dictionaries/page";
import BookmarksPage from "./routes/bookmarks/page";
import { InteractionCircuitPage } from "./routes/interaction-circuit/page";


const router = createBrowserRouter([
  {
    path: "/features",
    element: <FeaturesPage />,
  },
  {
    path: "/dictionaries",
    element: <DictionaryPage />,
  },
  {
    path: "/bookmarks",
    element: <BookmarksPage />,
  },
  {
    path: "/interaction-circuit",
    element: <InteractionCircuitPage />,
  },
  {
    path: "/",
    element: <RootPage />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
