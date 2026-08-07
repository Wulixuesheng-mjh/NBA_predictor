import { Routes, Route } from "react-router-dom";
import AppHeader from "./components/layout/AppHeader";
import HomePage from "./pages/HomePage";
import GamesPage from "./pages/GamesPage";
import GameDetailPage from "./pages/GameDetailPage";
import TeamsPage from "./pages/TeamsPage";
import TeamDetailPage from "./pages/TeamDetailPage";
import PlayerDetailPage from "./pages/PlayerDetailPage";
import HistoryPage from "./pages/HistoryPage";
import AdminPage from "./pages/AdminPage";

function App() {
  return (
    <div className="min-h-screen bg-gray-950">
      <AppHeader />
      <main className="max-w-7xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/games" element={<GamesPage />} />
          <Route path="/games/:gameId" element={<GameDetailPage />} />
          <Route path="/teams" element={<TeamsPage />} />
          <Route path="/teams/:teamId" element={<TeamDetailPage />} />
          <Route path="/players/:playerId" element={<PlayerDetailPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
