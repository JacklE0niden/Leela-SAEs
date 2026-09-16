import { cn } from "@/lib/utils";
import { Link, useLocation } from "react-router-dom";

const NAV_ITEMS = [
  ["/features", "Features"],
  ["/dictionaries", "Dictionaries"],
  ["/bookmarks", "Bookmarks"],
  ["/circuits", "Circuits"],
  ["/play-game", "Play Game"],
  ["/semantic-supernode-graph", "Semantic Supernode Graph"],
  ["/interaction-circuit", "Interaction Circuit"],
] as const;

export const AppNavbar = () => {
  const location = useLocation();

  return (
    <nav className="border-b bg-background p-4">
      <div className="container mx-auto flex items-center gap-6">
        <Link to="/play-game" aria-label="Circuit tracing home">
          <img src="/openmoss.ico" alt="logo" className="h-8" />
        </Link>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          {NAV_ITEMS.map(([to, label]) => (
            <Link
              key={to}
              className={cn(
                "whitespace-nowrap text-sm text-foreground/60 transition-colors hover:text-foreground/80",
                location.pathname === to && "font-medium text-foreground",
              )}
              to={to}
            >
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
};
