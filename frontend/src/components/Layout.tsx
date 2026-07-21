import { NavLink, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './Layout.css';

export function Layout() {
  const { role, email, logout } = useAuth();

  return (
    <div className="shell">
      <div className="prototype-banner">
        <strong>PROTOTYPE</strong> — not a medical device, not for clinical use. Corpus is real FDA
        drug labeling (openFDA), a small demo-scale subset — not a comprehensive or current reference.
      </div>

      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="font-display brand-name">SetuHealth</span>
        </div>

        <nav className="app-nav">
          <NavLink to="/ask" className={({ isActive }) => (isActive ? 'active' : '')}>
            Ask
          </NavLink>
          <NavLink to="/sources" className={({ isActive }) => (isActive ? 'active' : '')}>
            Sources
          </NavLink>
          {role === 'admin' && (
            <NavLink to="/admin" className={({ isActive }) => (isActive ? 'active' : '')}>
              Admin
            </NavLink>
          )}
        </nav>

        <div className="account">
          <span className="account-email">{email}</span>
          <span className="role-pill">{role}</span>
          <button type="button" className="link-button" onClick={logout}>
            Log out
          </button>
        </div>
      </header>

      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
