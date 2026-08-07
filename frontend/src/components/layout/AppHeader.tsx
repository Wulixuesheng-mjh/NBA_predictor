import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "首页" },
  { to: "/games", label: "赛程" },
  { to: "/teams", label: "球队" },
  { to: "/history", label: "历史" },
  { to: "/admin", label: "管理" },
];

export default function AppHeader() {
  return (
    <header className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <NavLink to="/" className="flex items-center gap-3">
          <span className="text-xl font-bold text-white tracking-tight">
            🏀 NBA Predictor
          </span>
        </NavLink>

        <nav className="flex items-center gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-blue-600 text-white"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
