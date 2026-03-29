import { cn } from "@/lib/utils";
import { Link, useLocation } from "react-router-dom";

export const AppNavbar = () => {
  const location = useLocation();
  const navItems = [
    { to: "/features", label: "Features" },
    { to: "/dictionaries", label: "Dictionaries" },
    { to: "/bookmarks", label: "Bookmarks" },
    { to: "/interaction-circuit", label: "Interaction Circuit" },
  ];

  return (
    <nav className="p-4">
      <div className="container mx-auto flex items-center gap-8">
        <img src="/openmoss.ico" alt="logo" className="h-8" />

        <div className="flex gap-4 items-center">
          {navItems.map((item) => (
            <Link
              key={item.to}
              className={cn(
                "transition-colors hover:text-foreground/80 text-foreground/60",
                location.pathname === item.to && "text-foreground"
              )}
              to={item.to}
            >
              {item.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
};
